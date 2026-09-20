// Steering ladder in one process: load the model once, then generate from every
// prompt at every coefficient.
//
// The coefficient multiplies the raw difference vector and the product is added
// to the residual stream after one decoder layer, at every token position --
// the same operation as the forward hook in scripts/4.2_steering/01, and the
// same one llama-cli performs for --control-vector-scaled FILE:COEFF. Doing it
// here avoids reloading 8 GB of weights once per rung of the ladder.
//
// The vector file is: int32 n_embd, then n_embd float32.
// Output is JSONL: {"coeff":..,"layer":..,"prompt_idx":..,"prompt":..,"completion":..}

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<float> read_vector(const std::string & path, int & n_embd) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "error: cannot open %s\n", path.c_str());
        exit(1);
    }
    int32_t n = 0;
    f.read((char *) &n, sizeof(n));
    std::vector<float> v(n);
    f.read((char *) v.data(), sizeof(float) * n);
    n_embd = n;
    return v;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += (char) c;
        }
    }
    return o;
}

static std::string piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    return n < 0 ? std::string() : std::string(buf, n);
}

static void usage(const char * exe) {
    fprintf(stderr,
        "usage: %s -m MODEL.gguf -f PROMPTS.txt --vector VEC.bin --layer L "
        "--coeffs \"-2,-1,0,1\" -o OUT.jsonl [options]\n"
        "  -n N     tokens to generate per prompt (default 120)\n"
        "  -ngl N   layers to offload to the GPU (default 0)\n"
        "  -c N     context size (default 1024)\n"
        "Generation is greedy, so a rung is fully determined by its coefficient.\n", exe);
}

int main(int argc, char ** argv) {
    std::string model_path, prompts_path, vector_path, out_path, coeffs_arg;
    int layer = -1, n_predict = 120, n_gpu_layers = 0, n_ctx = 1024;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "-m"       && i + 1 < argc) model_path   = argv[++i];
        else if (a == "-f"       && i + 1 < argc) prompts_path = argv[++i];
        else if (a == "--vector" && i + 1 < argc) vector_path  = argv[++i];
        else if (a == "--layer"  && i + 1 < argc) layer        = atoi(argv[++i]);
        else if (a == "--coeffs" && i + 1 < argc) coeffs_arg   = argv[++i];
        else if (a == "-o"       && i + 1 < argc) out_path     = argv[++i];
        else if (a == "-n"       && i + 1 < argc) n_predict    = atoi(argv[++i]);
        else if (a == "-ngl"     && i + 1 < argc) n_gpu_layers = atoi(argv[++i]);
        else if (a == "-c"       && i + 1 < argc) n_ctx        = atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompts_path.empty() || vector_path.empty() ||
        out_path.empty() || coeffs_arg.empty() || layer < 1) {
        usage(argv[0]);
        return 1;
    }

    std::vector<float> coeffs;
    { std::stringstream ss(coeffs_arg); std::string item;
      while (std::getline(ss, item, ',')) coeffs.push_back(std::stof(item)); }

    std::vector<std::string> prompts;
    { std::ifstream f(prompts_path); std::string line;
      while (std::getline(f, line)) {
          if (!line.empty() && line.back() == '\r') line.pop_back();
          if (!line.empty()) prompts.push_back(line);
      } }

    int vec_n_embd = 0;
    std::vector<float> direction = read_vector(vector_path, vec_n_embd);

    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * text, void *) {
        if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "error: failed to load %s\n", model_path.c_str()); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd   = llama_model_n_embd(model);
    const int n_layers = llama_model_n_layer(model);
    if (vec_n_embd != n_embd) {
        fprintf(stderr, "error: vector is %d wide, model is %d\n", vec_n_embd, n_embd);
        return 1;
    }
    if (layer >= n_layers) {
        fprintf(stderr, "error: layer %d out of range (model has %d)\n", layer, n_layers);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = n_ctx;
    cparams.n_batch  = n_ctx;
    cparams.n_ubatch = n_ctx;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "error: failed to create context\n"); return 1; }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // The buffer holds layers 1..layer; only the last slot is non-zero, so the
    // vector lands on exactly one layer just like the paper's single hook.
    std::vector<float> cvec_buf((size_t) n_embd * layer, 0.0f);

    std::ofstream out(out_path);
    std::vector<llama_token> tokens(n_ctx);

    for (float coeff : coeffs) {
        for (int k = 0; k < n_embd; k++) {
            cvec_buf[(size_t) n_embd * (layer - 1) + k] = coeff * direction[k];
        }
        if (llama_set_adapter_cvec(ctx, cvec_buf.data(), cvec_buf.size(), n_embd, layer, layer)) {
            fprintf(stderr, "error: failed to apply the control vector\n");
            return 1;
        }

        for (size_t p = 0; p < prompts.size(); p++) {
            int n_tok = llama_tokenize(vocab, prompts[p].c_str(), prompts[p].size(),
                                       tokens.data(), n_ctx, false, false);
            if (n_tok < 0) { fprintf(stderr, "error: prompt %zu too long\n", p); return 1; }

            llama_memory_clear(llama_get_memory(ctx), true);
            llama_sampler_reset(smpl);

            std::string completion;
            int n_past = 0;
            std::vector<llama_token> batch_tokens(tokens.begin(), tokens.begin() + n_tok);

            for (int t = 0; t < n_predict; t++) {
                if (llama_decode(ctx, llama_batch_get_one(batch_tokens.data(), batch_tokens.size()))) {
                    fprintf(stderr, "error: decode failed\n");
                    return 1;
                }
                n_past += batch_tokens.size();

                llama_token next = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, next) || n_past + 1 >= n_ctx) break;
                completion += piece(vocab, next);
                batch_tokens.assign(1, next);
            }

            out << "{\"coeff\":" << coeff
                << ",\"layer\":" << layer
                << ",\"prompt_idx\":" << p
                << ",\"prompt\":\"" << json_escape(prompts[p]) << "\""
                << ",\"completion\":\"" << json_escape(completion) << "\"}\n";
            out.flush();

            fprintf(stderr, "\rcoeff %+g  prompt %zu/%zu   ", coeff, p + 1, prompts.size());
            fflush(stderr);
        }
    }
    fprintf(stderr, "\nwrote %s\n", out_path.c_str());

    out.close();
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

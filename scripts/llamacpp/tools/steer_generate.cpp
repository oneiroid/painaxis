// Steering ladder in one process: load the model once, then generate from every
// prompt at every coefficient.
//
// Two modes:
//
//   add       h += coeff * v, at every token position after one decoder layer.
//             The same operation as the forward hook in scripts/4.2_steering/01
//             and as llama-cli's --control-vector-scaled FILE:COEFF. Applied
//             through llama.cpp's own control vector path.
//
//   geodesic  h -> |h| * normalize(h + coeff * v). Same direction of travel,
//             but the residual stream keeps the length it had. The point lands
//             on the great circle through h and v, which is the sphere analogue
//             of the straight-line shift -- the move used in GAN latent spaces
//             where the latent norm carries meaning. Applied by writing the
//             layer output back during the graph callback, since llama.cpp's
//             control vector path can only add.
//
// Either way the model is loaded once for the whole ladder.
//
// The vector file is: int32 n_embd, then n_embd float32.
// Output is JSONL: {"coeff":..,"layer":..,"prompt_idx":..,"prompt":..,"completion":..}

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// State for the geodesic mode, read by the graph callback.
struct geo_state {
    bool   active  = false;
    int    layer   = -1;
    int    n_embd  = 0;
    float  coeff   = 0.0f;
    std::vector<float> dir;    // the steering vector, unscaled
    std::vector<float> buf;    // scratch for one ubatch of the layer output
};

// Called for every graph node. On the target layer's output we read the tensor
// back, rotate each token's residual onto the sphere of its own radius, and
// write it back before the next node consumes it.
static bool cb_geodesic(ggml_tensor * t, bool ask, void * user_data) {
    auto * g = (geo_state *) user_data;

    const bool is_l_out = strncmp(t->name, "l_out-", 6) == 0;
    if (ask) {
        return is_l_out;
    }
    if (!g->active || !is_l_out || t->ne[0] != g->n_embd) {
        return true;
    }
    if (atoi(t->name + 6) != g->layer) {
        return true;
    }

    const int64_t n_tok = t->ne[1];
    g->buf.resize((size_t) g->n_embd * n_tok);
    ggml_backend_tensor_get(t, g->buf.data(), 0, ggml_nbytes(t));

    for (int64_t j = 0; j < n_tok; j++) {
        float * h = g->buf.data() + (size_t) j * g->n_embd;

        double r2 = 0.0;
        for (int k = 0; k < g->n_embd; k++) {
            r2 += (double) h[k] * h[k];
        }
        const double r = sqrt(r2);

        double s2 = 0.0;
        for (int k = 0; k < g->n_embd; k++) {
            const double x = h[k] + (double) g->coeff * g->dir[k];
            s2 += x * x;
        }
        const double scale = r / (sqrt(s2) + 1e-9);

        for (int k = 0; k < g->n_embd; k++) {
            h[k] = (float) (scale * (h[k] + (double) g->coeff * g->dir[k]));
        }
    }

    ggml_backend_tensor_set(t, g->buf.data(), 0, ggml_nbytes(t));
    return true;
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
        "  --mode M add (default) or geodesic; geodesic preserves the residual norm\n"
        "Generation is greedy, so a rung is fully determined by its coefficient.\n", exe);
}

int main(int argc, char ** argv) {
    std::string model_path, prompts_path, vector_path, out_path, coeffs_arg;
    std::string mode = "add";
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
        else if (a == "--mode"   && i + 1 < argc) mode         = argv[++i];
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompts_path.empty() || vector_path.empty() ||
        out_path.empty() || coeffs_arg.empty() || layer < 1) {
        usage(argv[0]);
        return 1;
    }
    if (mode != "add" && mode != "geodesic") {
        fprintf(stderr, "error: --mode must be add or geodesic\n");
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

    geo_state geo;
    geo.layer  = layer;
    geo.n_embd = n_embd;
    geo.dir    = direction;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = n_ctx;
    cparams.n_batch  = n_ctx;
    cparams.n_ubatch = n_ctx;
    if (mode == "geodesic") {
        cparams.cb_eval           = cb_geodesic;
        cparams.cb_eval_user_data = &geo;
    }
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
        if (mode == "geodesic") {
            geo.coeff  = coeff;
            geo.active = coeff != 0.0f;
        } else {
            for (int k = 0; k < n_embd; k++) {
                cvec_buf[(size_t) n_embd * (layer - 1) + k] = coeff * direction[k];
            }
            if (llama_set_adapter_cvec(ctx, cvec_buf.data(), cvec_buf.size(), n_embd, layer, layer)) {
                fprintf(stderr, "error: failed to apply the control vector\n");
                return 1;
            }
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
                << ",\"mode\":\"" << mode << "\""
                << ",\"layer\":" << layer
                << ",\"prompt_idx\":" << p
                << ",\"prompt\":\"" << json_escape(prompts[p]) << "\""
                << ",\"completion\":\"" << json_escape(completion) << "\"}\n";
            out.flush();

            fprintf(stderr, "\r%s coeff %+g  prompt %zu/%zu   ",
                    mode.c_str(), coeff, p + 1, prompts.size());
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

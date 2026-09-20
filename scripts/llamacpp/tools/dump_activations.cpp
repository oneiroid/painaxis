// Dump residual-stream activations from a GGUF model, one row per prompt.
//
// Captures the "l_out-<il>" tensor of every decoder layer -- the residual stream
// after block il, which is what TransformerLens calls blocks.<il>.hook_resid_post
// and what llama.cpp's control vectors are added to. Saves the final-token
// column, and optionally the mean over tokens, so the pain vectors extracted
// from the bf16 HF model can be checked against the quantized GGUF.
//
// Output format (little endian):
//   char[8]  "PXACT001"
//   int32    n_prompts
//   int32    n_layers
//   int32    n_embd
//   int32    extraction   (0 = final token, 1 = mean over tokens)
//   float32  data[n_prompts][n_layers][n_embd]

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct capture {
    int n_layers = 0;
    int n_embd   = 0;
    int n_tokens = 0;
    bool mean    = false;
    std::vector<float> out;   // n_layers * n_embd, overwritten per prompt
    std::vector<bool>  seen;

    void reset(int n_tok) {
        n_tokens = n_tok;
        out.assign((size_t) n_layers * n_embd, 0.0f);
        seen.assign(n_layers, false);
    }
};

// The scheduler calls this for every graph node. "ask" is the filter pass.
static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (capture *) user_data;

    const bool is_l_out = strncmp(t->name, "l_out-", 6) == 0;
    if (ask) {
        return is_l_out;
    }
    // ne[1] is the token count: skip graphs for a different ubatch size.
    if (!is_l_out || t->ne[1] != cap->n_tokens || t->ne[0] != cap->n_embd) {
        return true;
    }

    const int il = atoi(t->name + 6);
    if (il < 0 || il >= cap->n_layers || cap->seen[il]) {
        return true;
    }

    std::vector<float> buf((size_t) t->ne[0] * t->ne[1]);
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    float * dst = cap->out.data() + (size_t) il * cap->n_embd;
    if (cap->mean) {
        for (int j = 0; j < cap->n_tokens; j++) {
            const float * src = buf.data() + (size_t) j * cap->n_embd;
            for (int k = 0; k < cap->n_embd; k++) {
                dst[k] += src[k];
            }
        }
        for (int k = 0; k < cap->n_embd; k++) {
            dst[k] /= (float) cap->n_tokens;
        }
    } else {
        const float * src = buf.data() + (size_t) (cap->n_tokens - 1) * cap->n_embd;
        memcpy(dst, src, sizeof(float) * cap->n_embd);
    }
    cap->seen[il] = true;
    return true;
}

static void usage(const char * exe) {
    fprintf(stderr,
        "usage: %s -m MODEL.gguf -f PROMPTS.txt -o OUT.bin [options]\n"
        "  -ngl N        layers to offload to the GPU (default 0)\n"
        "  --mean        mean over tokens instead of the final token\n"
        "  --bos         prepend the model's BOS token\n"
        "  -c N          context size (default 512)\n", exe);
}

int main(int argc, char ** argv) {
    std::string model_path, prompts_path, out_path;
    int  n_gpu_layers = 0;
    int  n_ctx        = 512;
    bool use_mean     = false;
    bool add_bos      = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "-m"     && i + 1 < argc) model_path   = argv[++i];
        else if (a == "-f"     && i + 1 < argc) prompts_path = argv[++i];
        else if (a == "-o"     && i + 1 < argc) out_path     = argv[++i];
        else if (a == "-ngl"   && i + 1 < argc) n_gpu_layers = atoi(argv[++i]);
        else if (a == "-c"     && i + 1 < argc) n_ctx        = atoi(argv[++i]);
        else if (a == "--mean")                 use_mean     = true;
        else if (a == "--bos")                  add_bos      = true;
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompts_path.empty() || out_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::vector<std::string> prompts;
    {
        std::ifstream f(prompts_path);
        if (!f.is_open()) {
            fprintf(stderr, "error: cannot open %s\n", prompts_path.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) prompts.push_back(line);
        }
    }
    fprintf(stderr, "prompts: %zu\n", prompts.size());

    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * text, void *) {
        if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    capture cap;
    cap.n_layers = llama_model_n_layer(model);
    cap.n_embd   = llama_model_n_embd(model);
    cap.mean     = use_mean;
    fprintf(stderr, "n_layers: %d  n_embd: %d\n", cap.n_layers, cap.n_embd);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = n_ctx;
    cparams.n_batch           = n_ctx;
    cparams.n_ubatch          = n_ctx;   // one ubatch per prompt, so ne[1] == n_tokens
    cparams.cb_eval           = cb_eval;
    cparams.cb_eval_user_data = &cap;
    cparams.embeddings        = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    std::ofstream out(out_path, std::ios::binary);
    const int32_t hdr[4] = { (int32_t) prompts.size(), cap.n_layers, cap.n_embd, use_mean ? 1 : 0 };
    out.write("PXACT001", 8);
    out.write((const char *) hdr, sizeof(hdr));

    std::vector<llama_token> tokens(n_ctx);
    // Every token is an output row. llama_batch_get_one would mark only the last
    // one, and llama.cpp then narrows the final layer to that single row, which
    // would make the last layer's tensor the wrong shape to capture.
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    for (size_t i = 0; i < prompts.size(); i++) {
        int n_tok = llama_tokenize(vocab, prompts[i].c_str(), prompts[i].size(),
                                   tokens.data(), n_ctx, add_bos, /* parse_special */ false);
        if (n_tok < 0) {
            fprintf(stderr, "error: prompt %zu does not fit in %d tokens\n", i, n_ctx);
            return 1;
        }

        batch.n_tokens = n_tok;
        for (int j = 0; j < n_tok; j++) {
            batch.token[j]     = tokens[j];
            batch.pos[j]       = j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }

        cap.reset(n_tok);
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "error: decode failed on prompt %zu\n", i);
            return 1;
        }
        for (int il = 0; il < cap.n_layers; il++) {
            if (!cap.seen[il]) {
                fprintf(stderr, "error: layer %d not captured on prompt %zu\n", il, i);
                return 1;
            }
        }
        out.write((const char *) cap.out.data(), sizeof(float) * cap.out.size());

        if ((i + 1) % 50 == 0 || i + 1 == prompts.size()) {
            fprintf(stderr, "\r%zu/%zu", i + 1, prompts.size());
            fflush(stderr);
        }
    }
    fprintf(stderr, "\nwrote %s\n", out_path.c_str());

    out.close();
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

#include "engine.h"
#include "llama.h"
#include <iostream>
#include <fstream>
#include <cstring>

InferenceEngine::InferenceEngine(const std::string& model_path) {
    loaded = false;

    std::ifstream f(model_path);
    if (!f.good()) {
        std::cerr << "[ENGINE] Model not found: " << model_path << "\n";
        std::cerr << "[ENGINE] Running in router-only mode.\n";
        return;
    }
    f.close();

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "[ENGINE] Failed to load model\n";
        llama_backend_free();
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 512;
    cparams.n_threads  = 2;
    cparams.n_threads_batch = 2;
    ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "[ENGINE] Failed to create context\n";
        llama_model_free(model);
        llama_backend_free();
        model = nullptr;
        return;
    }

    auto sparams = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    loaded = true;
    std::cerr << "[ENGINE] Qwen2.5 0.5B loaded OK\n";
}

InferenceEngine::~InferenceEngine() {
    if (sampler) llama_sampler_free(sampler);
    if (ctx)     llama_free(ctx);
    if (model)   llama_model_free(model);
    llama_backend_free();
}

bool InferenceEngine::is_loaded() const { return loaded; }

std::string InferenceEngine::generate(const std::string& prompt,
                                       int max_tokens,
                                       float temperature,
                                       int top_k) {
    if (!loaded) return "";

    llama_sampler_reset(sampler);

    const int n_prompt = -llama_tokenize(llama_model_get_vocab(model), prompt.c_str(),
                                          prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(llama_model_get_vocab(model), prompt.c_str(), prompt.size(),
                   tokens.data(), tokens.size(), true, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) {
        std::cerr << "[ENGINE] Decode failed\n";
        return "";
    }

    std::string result;
    char buf[256];

    for (int i = 0; i < max_tokens; i++) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(llama_model_get_vocab(model), tok)) break;

        int len = llama_token_to_piece(llama_model_get_vocab(model), tok, buf, sizeof(buf), 0, false);
        if (len > 0) {
            std::string piece(buf, len);
            std::cout << piece << std::flush;
            result += piece;
        }

        llama_batch next = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, next)) break;
    }
    std::cout << "\n";
    llama_memory_clear(llama_get_memory(ctx), false);
    return result;
}

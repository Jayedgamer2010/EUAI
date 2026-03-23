#pragma once
#include <string>

struct llama_model;
struct llama_context;
struct llama_sampler;

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path);
    ~InferenceEngine();
    std::string generate(const std::string& prompt,
                         int max_tokens = 200,
                         float temperature = 0.8f,
                         int top_k = 40);
    bool is_loaded() const;
private:
    llama_model*   model   = nullptr;
    llama_context* ctx     = nullptr;
    llama_sampler* sampler = nullptr;
    bool loaded = false;
};

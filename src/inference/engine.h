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
    std::string model_path;
    bool loaded;
    std::string escape_for_shell(const std::string& s);
};

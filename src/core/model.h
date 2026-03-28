#pragma once

#include "matrix.h"
#include "attention.h"
#include <string>
#include <vector>
#include <cstdint>

struct ModelConfig {
    int dim;
    int n_layers;
    int n_heads;
    int n_kv_groups;
    int vocab_size;
    int max_seq_len;
    float rope_theta;
};

class Model {
public:
    Model();
    ~Model();

    bool load(const std::string& path);
    Matrix forward(const Matrix& tokens, int max_new_tokens = 200);

    int vocabSize() const { return config.vocab_size; }
    const ModelConfig& getConfig() const { return config; }

private:
    ModelConfig config;

    // Quantized weights storage (INT4 packed)
    struct QuantizedTensor {
        std::vector<int8_t> data;    // packed INT4 data (2 values per byte)
        std::vector<float> scales;   // per-block scales for dequantization
        std::vector<int> shape;      // [ndim, dim0, dim1, ...]
    };

    std::vector<QuantizedTensor> weights;
    std::vector<float*> layer_weights; // Dequantized on-demand

    // Components
    Attention* attention;

    // Helper functions
    Matrix rmsnorm(const Matrix& x, const float* weight);
    Matrix swiglu(const Matrix& x);
    Matrix layer_norm(const Matrix& x, const float* weight, const float* bias);
    Matrix matmul_dequantized(const Matrix& x, const QuantizedTensor& qw);
    Matrix softmax_last_dim(const Matrix& x);
    Matrix sample_token(const Matrix& logits, float temperature = 0.8f, int top_k = 40);
    void dequantize_layer_weights(int layer);

    // Load binary format
    bool load_binary(const std::string& path);
    bool load_gguf(const std::string& path);
};

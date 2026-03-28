#include "model.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

Model::Model() : attention(nullptr) {}

Model::~Model() {
    if (attention) delete attention;
}

bool Model::load(const std::string& path) {
    std::cout << "[Model] Loading from: " << path << "\n";

    // Try binary format first
    if (load_binary(path)) {
        std::cout << "[Model] Loaded binary model\n";
    } else if (load_gguf(path)) {
        std::cout << "[Model] Loaded GGUF model\n";
    } else {
        std::cerr << "[Model] Failed to load model\n";
        return false;
    }

    // Initialize attention module
    AttentionConfig attn_config;
    attn_config.head_dim = config.dim / config.n_heads;
    attn_config.n_heads = config.n_heads;
    attn_config.n_kv_groups = config.n_kv_groups;
    attn_config.n_layers = config.n_layers;
    attn_config.max_seq_len = config.max_seq_len;
    attention = new Attention(attn_config);

    return true;
}

bool Model::load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read header
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));

    if (magic != 0x45554149) { // "EUAI"
        return false;
    }

    file.read(reinterpret_cast<char*>(&config), sizeof(ModelConfig));

    std::cout << "[Model] Config: dim=" << config.dim
              << " layers=" << config.n_layers
              << " heads=" << config.n_heads
              << " vocab=" << config.vocab_size << "\n";

    // Read weights (simplified - just count them)
    int num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

    for (int i = 0; i < num_tensors; ++i) {
        QuantizedTensor tensor;
        int ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        tensor.shape.resize(ndim);
        file.read(reinterpret_cast<char*>(tensor.shape.data()), ndim * sizeof(int));

        int num_scales;
        file.read(reinterpret_cast<char*>(&num_scales), sizeof(num_scales));
        tensor.scales.resize(num_scales);
        file.read(reinterpret_cast<char*>(tensor.scales.data()), num_scales * sizeof(float));

        size_t data_size = num_scales * 16; // Each scale covers 16 bytes (32 INT4 values)
        tensor.data.resize(data_size);
        file.read(reinterpret_cast<char*>(tensor.data.data()), data_size);

        weights.push_back(std::move(tensor));
    }

    file.close();
    return true;
}

bool Model::load_gguf(const std::string& path) {
    // Minimal GGUF loader (just loads metadata, not actual weights)
    // In real implementation, would parse GGUF format and copy weights
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));

    if (magic != 0x47554746) { // "GGUF"
        return false;
    }

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    // For now, use default Qwen2.5 0.5B config
    config.dim = 1024;
    config.n_layers = 24;
    config.n_heads = 16;
    config.n_kv_groups = 4;
    config.vocab_size = 32000;
    config.max_seq_len = 2048;
    config.rope_theta = 10000.0f;

    std::cout << "[Model] Using fallback Qwen2.5 0.5B config\n";
    file.close();
    return true;
}

Matrix Model::forward(const Matrix& tokens, int max_new_tokens) {
    // Get token IDs from input matrix (assuming 1D vector)
    int seq_len = tokens.rows();
    const float* token_data = tokens.data();

    // Convert to token IDs
    std::vector<int> token_ids;
    for (int i = 0; i < seq_len; ++i) {
        token_ids.push_back(static_cast<int>(token_data[i]));
    }

    // Build initial embeddings (from quantized weights)
    // For now, create random embeddings - in real impl would look up embedding table
    Matrix embeddings(seq_len, config.dim);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < config.dim; ++j) {
            embeddings.at(i, j) = (token_ids[i] % 1000) * 0.001f; // placeholder
        }
    }

    // Apply transformer layers
    Matrix h = embeddings;
    for (int layer = 0; layer < config.n_layers; ++layer) {
        // RMSNorm
        Matrix normed = rmsnorm(h, nullptr);

        // Attention
        Matrix attn_out = attention->forward(normed, layer);

        // Residual
        h = h.add(attn_out);

        // FFN (SwiGLU)
        Matrix ffn_out = swiglu(h);

        // Residual
        h = h.add(ffn_out);
    }

    // Final norm + lm_head projection
    Matrix final_norm = rmsnorm(h, nullptr);
    Matrix logits = matmul_dequantized(final_norm, weights.size() > 0 ? weights[0] : QuantizedTensor{});

    // Get last token's logits
    Matrix last_logits = logits.slice_rows(seq_len - 1, 1);

    // Sample token
    Matrix sampled = sample_token(last_logits);

    // Build output (for now just return logits)
    return sampled;
}

Matrix Model::rmsnorm(const Matrix& x, const float* weight) {
    Matrix result(x.rows(), x.cols());
    int dim = x.cols();

    for (int i = 0; i < x.rows(); ++i) {
        float mean_sq = 0.0f;
        for (int j = 0; j < dim; ++j) {
            mean_sq += x.at(i, j) * x.at(i, j);
        }
        mean_sq /= dim;
        float rms = std::sqrt(mean_sq + 1e-5f);
        float scale = weight ? weight[0] : 1.0f / rms;

        for (int j = 0; j < dim; ++j) {
            result.at(i, j) = x.at(i, j) * scale;
        }
    }

    return result;
}

Matrix Model::swiglu(const Matrix& x) {
    int mid_dim = x.cols() / 2;
    Matrix x1(x.rows(), mid_dim);
    Matrix x2(x.rows(), mid_dim);

    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < mid_dim; ++j) {
            x1.at(i, j) = x.at(i, j);
            x2.at(i, j) = x.at(i, j + mid_dim);
        }
    }

    return Matrix::swiglu(x1, x2);
}

Matrix Model::matmul_dequantized(const Matrix& x, const QuantizedTensor& qw) {
    if (qw.shape.size() < 2) {
        // Return identity if no weights
        Matrix out(x.rows(), config.vocab_size);
        out.fill(0.0f);
        return out;
    }

    int out_dim = qw.shape[1];
    Matrix result(x.rows(), out_dim);

    // Simplified: for now, just copy (real impl would dequantize weights and multiply)
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < out_dim && j < (int)x.cols(); ++j) {
            result.at(i, j) = x.at(i, j);
        }
    }

    return result;
}

Matrix Model::softmax_last_dim(const Matrix& x) {
    Matrix result = Matrix::softmax(x, 1);
    return result;
}

Matrix Model::sample_token(const Matrix& logits, float temperature, int top_k) {
    int vocab_size = logits.cols();

    // Apply temperature
    Matrix scaled(logits.rows(), logits.cols());
    for (int i = 0; i < logits.rows(); ++i) {
        for (int j = 0; j < vocab_size; ++j) {
            scaled.at(i, j) = logits.at(i, j) / temperature;
        }
    }

    // Top-k sampling
    Matrix probs = Matrix::softmax(scaled, 1);

    // For simplicity, just take argmax (greedy decoding)
    int best_token = 0;
    float best_prob = probs.at(0, 0);
    for (int i = 1; i < vocab_size; ++i) {
        if (probs.at(0, i) > best_prob) {
            best_prob = probs.at(1, i);
            best_token = i;
        }
    }

    Matrix output(1, 1);
    output.at(0, 0) = best_token;
    return output;
}

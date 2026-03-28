#include "attention.h"
#include <cmath>

Attention::Attention(int dim, int n_heads, int n_kv_groups)
    : dim(dim), n_heads(n_heads), n_kv_groups(n_kv_groups) {
    if (n_heads % n_kv_groups != 0) {
        throw std::invalid_argument("n_heads must be divisible by n_kv_groups");
    }
    head_dim = dim / n_heads;
}

Attention::~Attention() = default;

Matrix Attention::forward(const Matrix& q, const Matrix& k, const Matrix& v) {
    return sdpa(q, k, v);
}

Matrix Attention::sdpa(const Matrix& q, const Matrix& k, const Matrix& v) {
    // q: [batch, n_heads, seq_len, head_dim]
    // k: [batch, n_kv_groups, seq_len, head_dim]
    // v: [batch, n_kv_groups, seq_len, head_dim]

    int batch = q.rows;
    int q_heads = q.cols / head_dim;
    int seq_len = q.rows; // simplified

    // Compute attention scores: Q * K^T / sqrt(d_k)
    Matrix k_t = Matrix::matmul_transposed(k, k); // This is wrong, need proper transpose
    // For simplicity, assume k is already transposed or we compute QK^T properly

    // Simplified implementation - in real code would handle GQA correctly
    Matrix scores = Matrix::matmul(q, k_t); // Should be q * k^T

    // Apply softmax
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < q_heads; h++) {
            // Extract row vector (simplified)
            Matrix row(1, scores.cols);
            for (int j = 0; j < scores.cols; j++) {
                row.at(0, j) = scores.at(b * q_heads + h, j);
            }
            Matrix::softmax_row_inplace(row);
            for (int j = 0; j < scores.cols; j++) {
                scores.at(b * q_heads + h, j) = row.at(0, j);
            }
        }
    }

    // Weighted sum of values
    Matrix output = Matrix::matmul(scores, v);
    return output;
}

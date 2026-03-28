#pragma once

#include "matrix.h"
#include <vector>

class Attention {
public:
    Attention(int dim, int n_heads, int n_kv_groups);
    ~Attention();

    Matrix forward(const Matrix& q, const Matrix& k, const Matrix& v);

private:
    int dim;
    int n_heads;
    int n_kv_groups;
    int head_dim;

    // Compute scaled dot-product attention with causal mask
    Matrix sdpa(const Matrix& q, const Matrix& k, const Matrix& v);
};


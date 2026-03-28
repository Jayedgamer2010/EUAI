#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

struct Matrix {
    int rows, cols;
    float* data;

    Matrix(int r, int c);
    Matrix(int r, int c, const float* init);
    ~Matrix();

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    float& at(int r, int c);
    const float& at(int r, int c) const;

    // Static operations return new matrices
    static Matrix matmul(const Matrix& a, const Matrix& b);
    static Matrix matmul_transposed(const Matrix& a, const Matrix& b);
    static void softmax_row_inplace(Matrix& m);
    static float silu(float x);
    static void rmsnorm(const Matrix& input, const float* weight, int size, float eps);
    static void copy_row(const Matrix& src, int row, float* dest);
};


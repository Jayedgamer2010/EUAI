#include "matrix.h"
#include <iostream>
#include <cstring>
#include <algorithm>

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    data = new float[r * c];
    std::memset(data, 0, r * c * sizeof(float));
}

Matrix::Matrix(int r, int c, const float* init) : rows(r), cols(c) {
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    data = new float[r * c];
    if (init) {
        std::memcpy(data, init, r * c * sizeof(float));
    } else {
        std::memset(data, 0, r * c * sizeof(float));
    }
}

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
    data = new float[rows * cols];
    std::memcpy(data, other.data, rows * cols * sizeof(float));
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        if (rows * cols != other.rows * other.cols) {
            delete[] data;
            rows = other.rows;
            cols = other.cols;
            data = new float[rows * cols];
        }
        std::memcpy(data, other.data, rows * cols * sizeof(float));
    }
    return *this;
}

Matrix::~Matrix() {
    delete[] data;
}

float& Matrix::at(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Row index out of bounds");
    }
    return data[r * cols + c];
}

const float& Matrix::at(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range("Row index out of bounds");
    }
    return data[r * cols + c];
}

Matrix Matrix::matmul(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix multiplication dimension mismatch");
    }
    Matrix result(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a.cols; k++) {
                sum += a.at(i, k) * b.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::matmul_transposed(const Matrix& a, const Matrix& b) {
    if (a.cols != b.cols) {
        throw std::invalid_argument("Matrix multiplication dimension mismatch");
    }
    Matrix result(a.rows, b.rows);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.rows; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a.cols; k++) {
                sum += a.at(i, k) * b.at(j, k);
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}

void Matrix::softmax_row_inplace(Matrix& m) {
    if (m.cols == 1) {
        // Special case for column vector
        float max_val = m.data[0];
        for (int i = 1; i < m.rows; i++) {
            if (m.data[i] > max_val) max_val = m.data[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < m.rows; i++) {
            m.data[i] = std::exp(m.data[i] - max_val);
            sum += m.data[i];
        }
        if (sum > 0) {
            for (int i = 0; i < m.rows; i++) {
                m.data[i] /= sum;
            }
        }
    } else if (m.rows == 1) {
        // Row vector
        float max_val = m.data[0];
        for (int j = 1; j < m.cols; j++) {
            if (m.data[j] > max_val) max_val = m.data[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < m.cols; j++) {
            m.data[j] = std::exp(m.data[j] - max_val);
            sum += m.data[j];
        }
        if (sum > 0) {
            for (int j = 0; j < m.cols; j++) {
                m.data[j] /= sum;
            }
        }
    } else {
        throw std::invalid_argument("softmax expects row or column vector");
    }
}

float Matrix::silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void Matrix::rmsnorm(const Matrix& input, const float* weight, int size, float eps) {
    if (input.cols != size) {
        throw std::invalid_argument("RMSNorm dimension mismatch");
    }
    for (int i = 0; i < input.rows; i++) {
        float sum_sq = 0.0f;
        for (int j = 0; j < size; j++) {
            float val = input.at(i, j);
            sum_sq += val * val;
        }
        float mean_sq = sum_sq / size;
        float rms = std::sqrt(mean_sq + eps);
        for (int j = 0; j < size; j++) {
            input.data[i * size + j] = (input.at(i, j) / rms) * weight[j];
        }
    }
}

void Matrix::copy_row(const Matrix& src, int row, float* dest) {
    if (row < 0 || row >= src.rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    std::memcpy(dest, src.data + row * src.cols, src.cols * sizeof(float));
}

// Additional helper for in-place addition
void add_inplace(Matrix& target, const Matrix& src) {
    if (target.rows != src.rows || target.cols != src.cols) {
        throw std::invalid_argument("Matrix addition dimension mismatch");
    }
    for (int i = 0; i < target.rows * target.cols; i++) {
        target.data[i] += src.data[i];
    }
}

// Additional helper for in-place scaling
void scale_inplace(Matrix& m, float scalar) {
    for (int i = 0; i < m.rows * m.cols; i++) {
        m.data[i] *= scalar;
    }
}

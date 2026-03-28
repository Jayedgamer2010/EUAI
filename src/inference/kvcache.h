#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>

struct KVLayer {
    float* k_cache;
    float* v_cache;
    size_t pos;
    size_t capacity;
    int head_dim;
};

class KVCache {
public:
    KVCache(int n_layers, int n_kv_groups, int head_dim, int max_seq_len);
    ~KVCache();

    void reset();
    void store(int layer, const float* k, const float* v, int seq_len);
    bool get(int layer, int pos, float* k_out, float* v_out) const;
    size_t current_pos(int layer) const;

private:
    std::vector<KVLayer> layers;
    int max_seq_len;
};


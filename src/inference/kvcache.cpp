#include "kvcache.h"
#include <cstring>
#include <iostream>

KVCache::KVCache(int n_layers, int n_kv_groups, int head_dim, int max_seq_len)
    : max_seq_len(max_seq_len) {
    layers.resize(n_layers * n_kv_groups);
    int total_size = head_dim * max_seq_len;

    for (auto& layer : layers) {
        layer.k_cache = new float[total_size];
        layer.v_cache = new float[total_size];
        layer.pos = 0;
        layer.capacity = total_size;
        layer.head_dim = head_dim;
        std::memset(layer.k_cache, 0, total_size * sizeof(float));
        std::memset(layer.v_cache, 0, total_size * sizeof(float));
    }
}

KVCache::~KVCache() {
    for (auto& layer : layers) {
        delete[] layer.k_cache;
        delete[] layer.v_cache;
    }
}

void KVCache::reset() {
    for (auto& layer : layers) {
        std::memset(layer.k_cache, 0, layer.capacity * sizeof(float));
        std::memset(layer.v_cache, 0, layer.capacity * sizeof(float));
        layer.pos = 0;
    }
}

void KVCache::store(int layer_idx, const float* k, const float* v, int seq_len) {
    if (layer_idx >= static_cast<int>(layers.size())) {
        return;
    }
    KVLayer& layer = layers[layer_idx];
    if (layer.pos + seq_len > max_seq_len) {
        std::cerr << "[KVCache] Overflow at layer " << layer_idx << "\n";
        return;
    }
    size_t offset = layer.pos * layer.head_dim;
    std::memcpy(layer.k_cache + offset, k, seq_len * layer.head_dim * sizeof(float));
    std::memcpy(layer.v_cache + offset, v, seq_len * layer.head_dim * sizeof(float));
    layer.pos += seq_len;
}

bool KVCache::get(int layer_idx, int pos, float* k_out, float* v_out) const {
    if (layer_idx >= static_cast<int>(layers.size())) return false;
    const KVLayer& layer = layers[layer_idx];
    if (pos >= static_cast<int>(layer.pos)) return false;
    size_t offset = pos * layer.head_dim;
    if (k_out) {
        std::memcpy(k_out, layer.k_cache + offset, layer.head_dim * sizeof(float));
    }
    if (v_out) {
        std::memcpy(v_out, layer.v_cache + offset, layer.head_dim * sizeof(float));
    }
    return true;
}

size_t KVCache::current_pos(int layer_idx) const {
    if (layer_idx >= static_cast<int>(layers.size())) return 0;
    return layers[layer_idx].pos;
}

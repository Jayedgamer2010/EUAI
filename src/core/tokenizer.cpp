#include "tokenizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

Tokenizer::Tokenizer(const std::string& vocab_path, const std::string& merges_path) {
    std::ifstream vf(vocab_path);
    if (vf.is_open()) {
        nlohmann::json v; vf >> v;
        for (auto& [t, id] : v.items()) {
            vocab[t] = id.get<int>();
            inv_vocab[id.get<int>()] = t;
        }
        std::cout << "[Tokenizer] loaded vocab " << vocab.size() << "\n";
    }

    std::ifstream mf(merges_path);
    if (mf.is_open()) {
        nlohmann::json m; mf >> m;
        int r = 0;
        for (auto& merge : m) {
            if (merge.is_array() && merge.size() >= 2) {
                std::string a = merge[0].get<std::string>();
                std::string b = merge[1].get<std::string>();
                if (vocab.count(a) && vocab.count(b)) {
                    merges.push_back({vocab[a], vocab[b], r++});
                }
            }
        }
        std::cout << "[Tokenizer] loaded merges " << merges.size() << "\n";
    }
}

Tokenizer::~Tokenizer() {}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    std::string word;
    for (char c : text) {
        if (std::isspace(c)) {
            if (!word.empty() && vocab.count(word)) {
                out.push_back(vocab.at(word));
            }
            word.clear();
        } else {
            word += c;
        }
    }
    if (!word.empty() && vocab.count(word)) {
        out.push_back(vocab.at(word));
    }
    if (out.empty()) out.push_back(0);
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string s;
    for (int t : tokens) {
        auto it = inv_vocab.find(t);
        if (it != inv_vocab.end()) s += it->second;
    }
    return s;
}

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    Tokenizer(const std::string& vocab_path, const std::string& merges_path);
    ~Tokenizer();

    std::vector<int> encode(const std::string& text) const;

    std::string decode(const std::vector<int>& tokens) const;

    int vocabSize() const { return vocab.size(); }

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    struct Merge { int left, right, rank; };
    std::vector<Merge> merges;
};

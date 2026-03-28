#pragma once
#include <string>
#include <vector>
#include <map>

struct KnowledgeEntry {
    std::string source_file;
    std::string content;
    std::vector<std::string> keywords;
};

class KnowledgeStore {
public:
    KnowledgeStore(const std::string& knowledge_dir);

    // Search returns best matching content or empty string
    std::string search(const std::string& query, float min_score = 0.4f);

    int entry_count() const { return entries.size(); }

private:
    std::vector<KnowledgeEntry> entries;

    void load_file(const std::string& path);
    void load_txt(const std::string& path);
    void load_json(const std::string& path);

    float score(const KnowledgeEntry& entry, const std::vector<std::string>& query_words);
    std::vector<std::string> tokenize(const std::string& text);
    std::string to_lower(const std::string& s);
};

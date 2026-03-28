#include "knowledge_store.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

KnowledgeStore::KnowledgeStore(const std::string& knowledge_dir) {
    try {
        if (!fs::exists(knowledge_dir)) {
            std::cerr << "[KnowledgeStore] Directory does not exist: " << knowledge_dir << "\n";
            return;
        }

        for (const auto& entry : fs::directory_iterator(knowledge_dir)) {
            if (!entry.is_regular_file()) continue;

            std::string path = entry.path().string();
            load_file(path);
        }

        std::cerr << "[KnowledgeStore] Loaded " << entries.size() << " knowledge entries\n";
    } catch (const std::exception& e) {
        std::cerr << "[KnowledgeStore] Error loading knowledge: " << e.what() << "\n";
    }
}

void KnowledgeStore::load_file(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    if (ext == ".txt") {
        load_txt(path);
    } else if (ext == ".json") {
        load_json(path);
    }
}

void KnowledgeStore::load_txt(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[KnowledgeStore] Failed to open: " << path << "\n";
            return;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        // Split by double newline into paragraphs
        std::istringstream iss(content);
        std::string paragraph;
        std::string accumulated;

        while (std::getline(iss, paragraph, '\n')) {
            if (paragraph.empty()) {
                if (!accumulated.empty()) {
                    KnowledgeEntry entry;
                    entry.source_file = path;
                    entry.content = accumulated;
                    entry.keywords = tokenize(accumulated);
                    entries.push_back(entry);
                    accumulated.clear();
                }
            } else {
                if (!accumulated.empty()) accumulated += "\n";
                accumulated += paragraph;
            }
        }

        // Don't forget the last paragraph
        if (!accumulated.empty()) {
            KnowledgeEntry entry;
            entry.source_file = path;
            entry.content = accumulated;
            entry.keywords = tokenize(accumulated);
            entries.push_back(entry);
        }

        std::cerr << "[KnowledgeStore] Loaded " << path << " with " << (entries.size()) << " entries\n";
    } catch (const std::exception& e) {
        std::cerr << "[KnowledgeStore] Error reading " << path << ": " << e.what() << "\n";
    }
}

void KnowledgeStore::load_json(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[KnowledgeStore] Failed to open: " << path << "\n";
            return;
        }

        nlohmann::json j;
        file >> j;

        if (j.is_array()) {
            // Array of objects with "content"
            for (const auto& item : j) {
                if (item.contains("content") && item["content"].is_string()) {
                    KnowledgeEntry entry;
                    entry.source_file = path;
                    entry.content = item["content"].get<std::string>();
                    entry.keywords = tokenize(entry.content);
                    entries.push_back(entry);
                }
            }
        } else if (j.is_string()) {
            // Treat as single text entry
            KnowledgeEntry entry;
            entry.source_file = path;
            entry.content = j.get<std::string>();
            entry.keywords = tokenize(entry.content);
            entries.push_back(entry);
        } else if (j.is_object()) {
            // Try to extract a "content" field from the object
            if (j.contains("content") && j["content"].is_string()) {
                KnowledgeEntry entry;
                entry.source_file = path;
                entry.content = j["content"].get<std::string>();
                entry.keywords = tokenize(entry.content);
                entries.push_back(entry);
            }
        }

        std::cerr << "[KnowledgeStore] Loaded " << path << " (total entries: " << entries.size() << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "[KnowledgeStore] Error parsing JSON " << path << ": " << e.what() << "\n";
    }
}

std::string KnowledgeStore::search(const std::string& query, float min_score) {
    if (entries.empty() || query.empty()) return "";

    // Tokenize query
    std::vector<std::string> query_words = tokenize(query);

    if (query_words.empty()) {
        // For very short queries (like "who am i"), try without stopword removal
        // Just get raw words, lowercase, no punctuation
        std::istringstream iss(query);
        std::string raw_word;
        while (iss >> raw_word) {
            std::string cleaned = to_lower(raw_word);
            cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(),
                                         [](char c) { return std::ispunct(c); }),
                          cleaned.end());
            if (!cleaned.empty() && cleaned.length() >= 1) {
                query_words.push_back(cleaned);
            }
        }
        if (query_words.empty()) return "";
    }

    // Score all entries
    KnowledgeEntry* best_entry = nullptr;
    float best_score = 0.0f;

    for (auto& entry : entries) {
        float s = score(entry, query_words);
        if (s > best_score) {
            best_score = s;
            best_entry = &entry;
        }
    }

    // Adjust threshold based on query length to avoid false positives
    float adjusted_min = min_score;
    if (query_words.size() == 1) {
        adjusted_min = 0.3f;  // Single word: lower threshold
    } else if (query_words.size() == 2) {
        adjusted_min = 0.6f;  // Two words: require both words to match (prevent generic single-word matches)
    } else {
        adjusted_min = 0.4f;  // Three+ words: moderate threshold
    }

    if (best_score >= adjusted_min && best_entry) {
        return best_entry->content;
    }

    return "";
}

float KnowledgeStore::score(const KnowledgeEntry& entry, const std::vector<std::string>& query_words) {
    if (query_words.empty()) return 0.0f;

    int matches = 0;
    for (const auto& qword : query_words) {
        for (const auto& kword : entry.keywords) {
            if (qword == kword) {
                matches++;
                break;
            }
        }
    }

    return static_cast<float>(matches) / query_words.size();
}

std::vector<std::string> KnowledgeStore::tokenize(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;

    // Stop words to ignore (keep pronouns for identity queries)
    static const std::unordered_set<std::string> stop_words = {
        "the", "a", "is", "in", "of", "to", "and", "for", "that", "this",
        "it", "with", "as", "on", "are", "was", "be", "at", "by", "from",
        "we", "they", "he", "she", "them", "his", "her",
        "have", "has", "had", "but", "not", "can", "will", "just", "my",
        "your", "our", "their", "its", "or", "if", "then", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "too", "very", "what", "which"
    };

    while (iss >> word) {
        // Convert to lowercase and remove punctuation
        std::string cleaned = to_lower(word);
        cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(),
                                     [](char c) { return std::ispunct(c); }),
                      cleaned.end());

        if (cleaned.empty()) continue;

        // Always keep 1-character words (pronouns like "i") and words > 2
        if (cleaned.length() == 1 || (cleaned.length() > 2 && stop_words.find(cleaned) == stop_words.end())) {
            words.push_back(cleaned);
        }
    }

    return words;
}

std::string KnowledgeStore::to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

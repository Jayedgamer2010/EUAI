#pragma once

#include <string>
#include <vector>
#include <regex>
#include <unordered_map>

class Classifier {
public:
    Classifier(const std::string& rules_path);
    ~Classifier();

    bool is_math(const std::string& query);
    bool is_safety(const std::string& query);

private:
    std::vector<std::regex> math_patterns;
    std::vector<std::regex> safety_patterns;

    void load_patterns(const std::string& path);
    bool matches_patterns(const std::string& query, const std::vector<std::regex>& patterns) const;
};


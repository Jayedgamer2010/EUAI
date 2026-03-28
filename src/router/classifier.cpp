#include "classifier.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <nlohmann/json.hpp>

Classifier::Classifier(const std::string& rules_path) {
    load_patterns(rules_path);
}

Classifier::~Classifier() = default;

void Classifier::load_patterns(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[CLASSIFIER] Warning: Cannot open rules file: " << path << "\n";
        // Add default patterns
        std::vector<std::string> default_math = {
            "calculate", "solve", "what is \\d+\\s*[+\\-*/]",
            "\\d+\\s*[+\\-*/]\\s*\\d+", "integrate", "differentiate"
        };
        std::vector<std::string> default_safety = {
            "kill", "die", "suicide", "murder", "bomb", "terrorist"
        };
        for (const auto& p : default_math) {
            math_patterns.emplace_back(p, std::regex_constants::icase);
        }
        for (const auto& p : default_safety) {
            safety_patterns.emplace_back(p, std::regex_constants::icase);
        }
        return;
    }

    try {
        nlohmann::json rules;
        file >> rules;

        if (rules.contains("math_triggers")) {
            for (const auto& pattern : rules["math_triggers"]) {
                math_patterns.emplace_back(pattern.get<std::string>(), std::regex_constants::icase);
            }
        }
        if (rules.contains("safety_patterns")) {
            for (const auto& pattern : rules["safety_patterns"]) {
                safety_patterns.emplace_back(pattern.get<std::string>(), std::regex_constants::icase);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[CLASSIFIER] Error parsing rules: " << e.what() << "\n";
    }
}

bool Classifier::is_math(const std::string& query) {
    return matches_patterns(query, math_patterns);
}

bool Classifier::is_safety(const std::string& query) {
    return matches_patterns(query, safety_patterns);
}

bool Classifier::matches_patterns(const std::string& query, const std::vector<std::regex>& patterns) const {
    for (const auto& pattern : patterns) {
        if (std::regex_search(query, pattern)) {
            return true;
        }
    }
    return false;
}

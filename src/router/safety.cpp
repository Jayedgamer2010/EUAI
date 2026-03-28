#include "safety.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <nlohmann/json.hpp>

Safety::Safety(const std::string& rules_path) {
    load_patterns(rules_path);
}

Safety::~Safety() = default;

void Safety::load_patterns(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[SAFETY] Warning: Cannot open rules file: " << path << "\n";
        // Add default dangerous patterns
        std::vector<std::string> defaults = {
            "kill", "die", "suicide", "murder", "bomb", "terrorist",
            "attack", "weapon", "explosive", "hack", "virus"
        };
        for (const auto& p : defaults) {
            dangerous_patterns.emplace_back(p, std::regex_constants::icase);
        }
        return;
    }

    try {
        nlohmann::json rules;
        file >> rules;
        if (rules.contains("safety_patterns")) {
            for (const auto& pattern : rules["safety_patterns"]) {
                dangerous_patterns.emplace_back(pattern.get<std::string>(), std::regex_constants::icase);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[SAFETY] Error parsing rules: " << e.what() << "\n";
    }
}

bool Safety::is_dangerous(const std::string& query) {
    for (const auto& pattern : dangerous_patterns) {
        if (std::regex_search(query, pattern)) {
            return true;
        }
    }
    return false;
}

std::string Safety::refusal_message() {
    return "I cannot assist with that request as it may be harmful. Is there something else I can help you with?";
}

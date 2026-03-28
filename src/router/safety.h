#pragma once

#include <string>
#include <vector>
#include <regex>

class Safety {
public:
    Safety(const std::string& rules_path);
    ~Safety();

    bool is_dangerous(const std::string& query);
    std::string refusal_message();

private:
    std::vector<std::regex> dangerous_patterns;
    void load_patterns(const std::string& path);
};


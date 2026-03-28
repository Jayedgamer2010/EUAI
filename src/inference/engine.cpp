#include "engine.h"
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>

InferenceEngine::InferenceEngine(const std::string& model_path) : loaded(false) {
    std::ifstream f(model_path);
    if (f.good()) {
        loaded = true;
        this->model_path = model_path;
        std::cerr << "[InferenceEngine] Model ready: " << model_path << "\n";
    } else {
        std::cerr << "[InferenceEngine] Model not found: " << model_path << "\n";
    }
}

InferenceEngine::~InferenceEngine() {}

bool InferenceEngine::is_loaded() const {
    return loaded;
}

std::string InferenceEngine::generate(const std::string& prompt,
                                     int max_tokens,
                                     float temperature,
                                     int top_k) {
    if (!loaded) return "[ERROR] Model not loaded";

    // Load system prompt from file
    std::string system_prompt = "You are EUAI. Answer concisely.";
    std::ifstream sp_file("config/system_prompt.txt");
    if (sp_file.good()) {
        std::string sp_content((std::istreambuf_iterator<char>(sp_file)),
                                std::istreambuf_iterator<char>());
        if (!sp_content.empty()) system_prompt = sp_content;
    }

    // Format prompt for Qwen2.5 chat template
    std::ostringstream oss;
    oss << "system\n" << system_prompt << "\n\n";
    oss << "user\n" << prompt << "\n\n";
    oss << "assistant";
    std::string formatted = oss.str();

    // Escape the prompt for safe shell execution
    std::string escaped = escape_for_shell(formatted);

    // Build command: pass prompt as argument to llama-simple
    // Use -p flag to explicitly set prompt
    // Added: repeat penalty, top_p, and better sampling to avoid loops
    // Note: llama.cpp uses underscores for some flags (--top_p, --repeat_penalty)
    std::string cmd = "/home/storage/EUAI/llama.cpp/build/bin/llama-simple -m " + model_path +
                      " -n " + std::to_string(max_tokens) +
                      " --temp " + std::to_string(temperature) +
                      " --top-k " + std::to_string(top_k) +
                      " --top_p 0.90" +
                      " --repeat_penalty 1.50" +
                      " --repeat_last_n 64" +
                      " -p " + escaped +
                      " 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "[ERROR] Failed to run llama-simple";

    char buf[4096];
    std::string output;
    while (fgets(buf, sizeof(buf), pipe)) {
        output += buf;
    }
    pclose(pipe);

    // Extract assistant response: find the assistant marker that is preceded by a newline or start of string
    // The marker is typically "\nassistant" or just "assistant" at the end of the prompt
    size_t assistant_pos = std::string::npos;
    size_t pos = output.find("\nassistant");
    if (pos != std::string::npos) {
        assistant_pos = pos + 1; // point to 'a' of assistant
    } else {
        pos = output.find("assistant");
        // Check if it's a standalone marker (preceded by newline or start, and followed by space or end)
        while (pos != std::string::npos) {
            if ((pos == 0 || output[pos-1] == '\n' || output[pos-1] == ' ') &&
                (pos + 9 >= output.size() || std::isspace(output[pos+9]))) {
                assistant_pos = pos;
                break;
            }
            pos = output.find("assistant", pos + 1);
        }
    }

    if (assistant_pos != std::string::npos) {
        output = output.substr(assistant_pos + 9); // skip "assistant"
    } else {
        // If no assistant marker found, return original trimmed (unlikely)
    }

    // Remove EOS token and everything after
    size_t eos_pos = output.find("<|endoftext|>");
    if (eos_pos != std::string::npos) {
        output.resize(eos_pos);
    }

    // Also check for "system" re-appearance (multi-turn issue)
    size_t system_pos = output.find("system\n");
    if (system_pos != std::string::npos) {
        output = output.substr(0, system_pos);
    }

    // Trim whitespace
    output.erase(0, output.find_first_not_of(" \t\n\r"));
    if (!output.empty()) {
        output.erase(output.find_last_not_of(" \t\n\r") + 1);
    }

    return output.empty() ? "[EMPTY RESPONSE]" : output;
}

// Helper: escape a string for safe inclusion in double-quoted shell argument
std::string InferenceEngine::escape_for_shell(const std::string& s) {
    std::string result = "\"";
    for (char c : s) {
        if (c == '\\' || c == '"') {
            result += '\\';
        }
        result += c;
    }
    result += "\"";
    return result;
}

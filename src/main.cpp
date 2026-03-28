#include <iostream>
#include <string>
#include <getopt.h>
#include <csignal>
#include <unistd.h> // for isatty

#include "router/router.h"
#include "core/tokenizer.h"

volatile std::sig_atomic_t g_signal_received = 0;

void signal_handler(int signal) {
    g_signal_received = signal;
}

void print_usage() {
    std::cout << "EUAI - Hybrid C++/Python Language Model\n\n";
    std::cout << "Usage: euai [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --model <path>    Path to GGUF model (default: models/qwen2.5-coder-0.5b-instruct-q2_k.gguf)\n";
    std::cout << "  --config <dir>    Config directory (default: config/)\n";
    std::cout << "  --max-tokens <n>  Maximum tokens to generate (default: 200)\n";
    std::cout << "  --stats           Print statistics on exit\n";
    std::cout << "  -h, --help        Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  ./euai --model python/euai.bin --config config/\n";
    std::cout << "  ./euai --max-tokens 100 --stats\n";
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/qwen2.5-coder-0.5b-instruct-q2_k.gguf";
    std::string config_dir = "config/";
    int max_tokens = 200;

    // Parse command line arguments
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"config", required_argument, 0, 'c'},
        {"max-tokens", required_argument, 0, 't'},
        {"stats", no_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    bool show_stats = false;
    while (true) {
        int option_index = 0;
        int c = getopt_long(argc, argv, "hm:c:t:s", long_options, &option_index);
        if (c == -1) break;

        switch (c) {
            case 'm':
                model_path = optarg;
                break;
            case 'c':
                config_dir = optarg;
                break;
            case 't':
                max_tokens = std::stoi(optarg);
                break;
            case 's':
                show_stats = true;
                break;
            case 'h':
                print_usage();
                return 0;
            default:
                print_usage();
                return 1;
        }
    }

    bool interactive = isatty(STDIN_FILENO);

    try {
        Router router(config_dir, model_path);

        if (interactive) {
            std::cerr << "[MAIN] Starting EUAI\n";
            std::cerr << "[MAIN] Model: " << model_path << "\n";
            std::cerr << "[MAIN] Config: " << config_dir << "\n";
            std::cerr << "\n=== EUAI Interactive Mode ===\n";
            std::cerr << "Type your query (or 'quit' to exit)\n";
            if (show_stats) {
                std::cerr << "(Stats will be printed on exit)\n";
            }
            std::cerr << "\n";
        }

        std::string line;
        while (true) {
            // Check for termination signal
            if (g_signal_received) {
                if (interactive) std::cerr << "\n[MAIN] Interrupt received, shutting down...\n";
                break;
            }

            // Check if stdin has closed (pipe ended)
            if (!std::cin.good() && std::cin.eof()) {
                break;
            }

            if (interactive) {
                std::cerr << "> ";
                std::cerr.flush();
            }

            if (!std::getline(std::cin, line)) {
                if (std::cin.eof()) break;
                std::cin.clear();
                continue;
            }

            if (line == "quit" || line == "exit") break;
            if (line.empty()) continue;

            try {
                std::string response = router.route(line);
                // In interactive mode, separate with newlines; in non-interactive, just output response
                if (interactive) {
                    std::cerr << response << "\n\n";
                } else {
                    std::cout << response << "\n";
                    std::cout.flush();
                }
            } catch (const std::exception& e) {
                if (interactive) {
                    std::cerr << "[ERROR] " << e.what() << "\n\n";
                } else {
                    std::cerr << "[ERROR] " << e.what() << "\n";
                }
            }
        }

        if (interactive) {
            std::cerr << "[MAIN] Shutting down\n";
        }

        if (show_stats) {
            router.print_stats(std::cerr);
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}

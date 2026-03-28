#include "router.h"
#include "classifier.h"
#include "safety.h"
#include "math_engine.h"
#include "../inference/engine.h"
#include "cache.h"
#include "knowledge_store.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <regex>
#include <mutex>

Router::Router(const std::string& config_dir, const std::string& model_path)
    : classifier(std::make_unique<Classifier>(config_dir + "/router_rules.json")),
      safety(std::make_unique<Safety>(config_dir + "/router_rules.json")),
      math_engine(std::make_unique<MathEngine>()),
      neural_engine(std::make_unique<InferenceEngine>(model_path)),
      knowledge_store(std::make_unique<KnowledgeStore>("knowledge/")) {

    // Load cache configuration
    std::time_t ttl_seconds = 3600; // default 1 hour
    try {
        std::ifstream file(config_dir + "/router_rules.json");
        if (file.is_open()) {
            nlohmann::json rules;
            file >> rules;
            if (rules.contains("cache_ttl_seconds")) {
                ttl_seconds = rules["cache_ttl_seconds"].get<std::time_t>();
            }
        }
    } catch (...) {
        // Use default
    }

    cache = std::make_unique<QueryCache>(1000, ttl_seconds);
}

Router::~Router() = default;

QueryType Router::classify(const std::string& query) {
    // Safety check first
    if (safety->is_dangerous(query)) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.safety_queries++;
        return QueryType::SAFETY;
    }

    // Check for math
    if (math_engine->can_handle(query)) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.math_queries++;
        return QueryType::MATH;
    }

    // Check cache (if we have a cached response)
    if (cache) {
        std::string cached = cache->get(query);
        if (!cached.empty()) {
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats.cache_queries++;
            stats.cache_hits++;
            return QueryType::CACHE;
        }
        // Cache miss but we'll route to neural or knowledge
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.cache_misses++;
    }

    // Check knowledge store
    if (knowledge_store) {
        std::string knowledge_result = knowledge_store->search(query);
        if (!knowledge_result.empty()) {
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats.knowledge_queries++;
            return QueryType::KNOWLEDGE;
        }
    }

    // Default to neural
    std::lock_guard<std::mutex> lock(stats_mutex);
    stats.neural_queries++;
    return QueryType::NEURAL;
}

std::string Router::route(const std::string& query) {
    QueryType type = classify(query);

    std::string response;
    switch (type) {
        case QueryType::SAFETY:
            response = safety->refusal_message();
            break;

        case QueryType::MATH:
            response = handle_math(query);
            break;

        case QueryType::CACHE:
            response = handle_cache(query);
            break;

        case QueryType::KNOWLEDGE:
            response = handle_knowledge(query);
            break;

        case QueryType::NEURAL:
            response = handle_neural(query);
            break;

        case QueryType::UNKNOWN:
        default:
            response = "I'm not sure how to respond to that.";
            break;
    }

    // Tag response with source
    std::string tag;
    switch (type) {
        case QueryType::MATH: tag = "[SOURCE: MATH] "; break;
        case QueryType::CACHE: tag = "[SOURCE: CACHE] "; break;
        case QueryType::NEURAL: tag = "[SOURCE: NEURAL] "; break;
        case QueryType::SAFETY: tag = "[SOURCE: SAFETY] "; break;
        case QueryType::KNOWLEDGE: tag = "[SOURCE: KNOWLEDGE] "; break;
        default: tag = "[SOURCE: UNKNOWN] ";
    }

    return tag + response;
}

std::string Router::handle_math(const std::string& expr) {
    return math_engine->solve(expr);
}

std::string Router::handle_cache(const std::string& query) {
    if (cache) {
        std::string cached = cache->get(query);
        if (!cached.empty()) {
            return cached;
        }
    }
    return "[CACHE] Miss: " + query;
}

std::string Router::handle_neural(const std::string& query) {
    if (!neural_engine->is_loaded()) {
        return "Neural engine not available. Please train and export a model first.";
    }

    // Check cache again (might have been populated by another request)
    if (cache) {
        std::string cached = cache->get(query);
        if (!cached.empty()) {
            return cached;
        }
    }

    // Generate fresh response
    std::string response = neural_engine->generate(query, 200, 0.8f, 40);

    // Cache successful non-empty responses
    if (cache && !response.empty() && response.find("[ERROR]") == std::string::npos) {
        cache->put(query, response);
    }

    return response;
}

std::string Router::handle_knowledge(const std::string& query) {
    if (knowledge_store) {
        return knowledge_store->search(query);
    }
    return "";
}

Router::Stats Router::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex);
    return stats;
}

void Router::print_stats(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(stats_mutex);
    os << "\n=== EUAI Statistics ===\n";
    os << "Neural queries: " << stats.neural_queries << "\n";
    os << "Math queries: " << stats.math_queries << "\n";
    os << "Safety queries: " << stats.safety_queries << "\n";
    os << "Knowledge queries: " << stats.knowledge_queries << "\n";
    os << "Cache hits: " << stats.cache_hits << "\n";
    os << "Cache misses: " << stats.cache_misses << "\n";
    if (cache) {
        os << "Cache size: " << cache->size() << " entries\n";
        os << "Cache total hits: " << cache->hits() << "\n";
        os << "Cache total misses: " << cache->misses() << "\n";
    }
    if (knowledge_store) {
        os << "Knowledge entries: " << knowledge_store->entry_count() << "\n";
    }
    os << "=======================\n";
}

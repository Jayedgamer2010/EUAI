#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <ostream>

class Classifier;
class Safety;
class MathEngine;
class InferenceEngine;
class QueryCache;
class KnowledgeStore;

enum class QueryType {
    MATH,
    CACHE,
    SAFETY,
    NEURAL,
    KNOWLEDGE,
    UNKNOWN
};

class Router {
public:
    Router(const std::string& config_dir, const std::string& model_path);
    ~Router();

    QueryType classify(const std::string& query);
    std::string route(const std::string& query);

    // Stats for monitoring
    struct Stats {
        int neural_queries = 0;
        int math_queries = 0;
        int safety_queries = 0;
        int cache_queries = 0;
        int knowledge_queries = 0;
        int cache_hits = 0;
        int cache_misses = 0;
    };
    Stats get_stats() const;
    void print_stats(std::ostream& os) const;

private:
    std::unique_ptr<class Classifier> classifier;
    std::unique_ptr<class Safety> safety;
    std::unique_ptr<class MathEngine> math_engine;
    std::unique_ptr<InferenceEngine> neural_engine;
    std::unique_ptr<QueryCache> cache;
    std::unique_ptr<KnowledgeStore> knowledge_store;

    std::string handle_math(const std::string& expr);
    std::string handle_cache(const std::string& query);
    std::string handle_neural(const std::string& query);
    std::string handle_knowledge(const std::string& query);

    // Internal stats
    mutable std::mutex stats_mutex;
    Stats stats;
};


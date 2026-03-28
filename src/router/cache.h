#pragma once

#include <string>
#include <unordered_map>
#include <list>
#include <chrono>
#include <mutex>

struct CacheEntry {
    std::string response;
    std::time_t timestamp;
    int hit_count;
};

class QueryCache {
public:
    QueryCache(size_t max_size = 1000, std::time_t ttl_seconds = 3600);
    ~QueryCache();

    // Get cached response, empty string if not found or expired
    std::string get(const std::string& query);

    // Store response in cache
    void put(const std::string& query, const std::string& response);

    // Clear expired entries and enforce size limit
    void cleanup();

    // Stats
    size_t size() const;
    size_t hits() const;
    size_t misses() const;

private:
    // Normalize query for consistent hashing/lookup
    std::string normalize(const std::string& query) const;

    // Data structures
    std::unordered_map<std::string, CacheEntry> cache;
    std::list<std::string> lru_order; // Most recent at front

    size_t max_size;
    std::time_t ttl_seconds;

    // Thread safety
    mutable std::mutex mutex;

    // Stats
    size_t hits_total = 0;
    size_t misses_total = 0;
};

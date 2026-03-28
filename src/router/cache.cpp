#include "cache.h"
#include <algorithm>
#include <cctype>
#include <time.h>

QueryCache::QueryCache(size_t max_size, std::time_t ttl_seconds)
    : max_size(max_size), ttl_seconds(ttl_seconds) {
}

QueryCache::~QueryCache() = default;

std::string QueryCache::normalize(const std::string& query) const {
    std::string norm = query;
    // Trim whitespace
    norm.erase(0, norm.find_first_not_of(" \t\n\r"));
    norm.erase(norm.find_last_not_of(" \t\n\r") + 1);
    // Convert to lowercase
    std::transform(norm.begin(), norm.end(), norm.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return norm;
}

std::string QueryCache::get(const std::string& query) {
    std::lock_guard<std::mutex> lock(mutex);

    std::string norm = normalize(query);
    if (norm.empty()) return "";

    auto it = cache.find(norm);
    if (it == cache.end()) {
        misses_total++;
        return "";
    }

    // Check expiration
    std::time_t now = std::time(nullptr);
    if (ttl_seconds > 0 && (now - it->second.timestamp) > ttl_seconds) {
        cache.erase(it);
        // Remove from LRU list
        lru_order.remove(norm);
        misses_total++;
        return "";
    }

    // Cache hit
    hits_total++;
    it->second.hit_count++;
    // Move to front of LRU
    lru_order.remove(norm);
    lru_order.push_front(norm);
    return it->second.response;
}

void QueryCache::put(const std::string& query, const std::string& response) {
    if (response.empty()) return; // Don't cache empty responses

    std::lock_guard<std::mutex> lock(mutex);

    std::string norm = normalize(query);
    if (norm.empty()) return;

    std::time_t now = std::time(nullptr);

    auto it = cache.find(norm);
    if (it != cache.end()) {
        // Update existing entry
        it->second.response = response;
        it->second.timestamp = now;
        it->second.hit_count++;
        lru_order.remove(norm);
        lru_order.push_front(norm);
        return;
    }

    // Insert new entry
    if (cache.size() >= max_size) {
        // Evict least recently used (back of list)
        if (!lru_order.empty()) {
            std::string evict = lru_order.back();
            lru_order.pop_back();
            cache.erase(evict);
        }
    }

    CacheEntry entry{response, now, 1};
    cache.emplace(norm, entry);
    lru_order.push_front(norm);
}

void QueryCache::cleanup() {
    std::lock_guard<std::mutex> lock(mutex);

    if (ttl_seconds <= 0) return;

    std::time_t now = std::time(nullptr);
    auto it = cache.begin();
    while (it != cache.end()) {
        if ((now - it->second.timestamp) > ttl_seconds) {
            lru_order.remove(it->first);
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

size_t QueryCache::size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return cache.size();
}

size_t QueryCache::hits() const {
    std::lock_guard<std::mutex> lock(mutex);
    return hits_total;
}

size_t QueryCache::misses() const {
    std::lock_guard<std::mutex> lock(mutex);
    return misses_total;
}

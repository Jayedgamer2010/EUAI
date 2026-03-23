#pragma once
#include <string>
#include <stdexcept>

struct common_http_url {
    std::string scheme, user, password, host, path;
    int port = 80;
};

static common_http_url common_http_parse_url(const std::string&) {
    throw std::runtime_error("HTTP not supported in this build");
}

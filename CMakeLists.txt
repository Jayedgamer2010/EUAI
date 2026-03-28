cmake_minimum_required(VERSION 3.16)
project(EUAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(SQLite3 REQUIRED sqlite3)
find_path(JSONCPP_INCLUDE_DIR nlohmann/json.hpp)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/router
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${JSONCPP_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp
)

# Source files
set(SOURCES
    src/core/matrix.cpp
    src/core/tokenizer.cpp
    src/core/model.cpp
    src/core/attention.cpp
    src/router/router.cpp
    src/router/classifier.cpp
    src/router/safety.cpp
    src/router/math_engine.cpp
    src/inference/kvcache.cpp
    src/inference/engine.cpp
    src/main.cpp
)

set(HEADERS
    src/core/matrix.h
    src/core/tokenizer.h
    src/core/model.h
    src/core/attention.h
    src/router/router.h
    src/router/classifier.h
    src/router/safety.h
    src/router/math_engine.h
    src/inference/kvcache.h
    src/inference/engine.h
)

# Llama.cpp object files
set(LLAMA_SOURCES
    llama.cpp/src/llama.cpp
    llama.cpp/src/llama-model.cpp
    llama.cpp/src/llama-arch.cpp
    llama.cpp/src/llama-context.cpp
    # Add more llama.cpp sources as needed
)

add_executable(euai ${SOURCES} ${HEADERS} ${LLAMA_SOURCES})

target_include_directories(euai PRIVATE
    llama.cpp/src
    llama.cpp/include
)

target_link_libraries(euai PRIVATE ${SQLite3_LIBRARIES} pthread dl)
target_compile_options(euai PRIVATE ${SQLite3_CFLAGS_OTHER})

# Test executable
add_executable(euai_test test/test_main.cpp ${SOURCES} ${LLAMA_SOURCES})
target_include_directories(euai_test PRIVATE
    llama.cpp/src
    llama.cpp/include
)
target_link_libraries(euai_test PRIVATE ${SQLite3_LIBRARIES} pthread dl)

# Installation
install(TARGETS euai DESTINATION bin)

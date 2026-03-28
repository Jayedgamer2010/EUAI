# EUAI Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O2 -I./src/core -I./src/router -I./src/inference -I/data/data/com.termux/files/usr/include
LDFLAGS = -lsqlite3 -lpthread -ldl

# Source files - only production-used components
# Note: model.cpp, attention.cpp, matrix.cpp are NOT used (llama-simple handles inference)
ROUTER_SRCS = src/router/router.cpp src/router/classifier.cpp src/router/safety.cpp src/router/math_engine.cpp src/router/cache.cpp
INF_SRCS = src/inference/engine.cpp
SRCS = $(ROUTER_SRCS) $(INF_SRCS) src/main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean test python-install run

all: euai

# Build using object files
euai: $(OBJS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o build/euai $^ $(LDFLAGS)
	@echo "Built euai binary at build/euai"

# Pattern rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f src/core/matrix.o src/core/tokenizer.o src/router/router.o src/router/classifier.o src/router/safety.o src/router/math_engine.o src/inference/kvcache.o src/inference/engine.o src/main.o
	rm -f build/euai build/euai_test
	rm -rf python/checkpoints/

test: euai_test
	./build/euai_test

# Exclude main.o for test build
TEST_OBJS = $(filter-out src/main.o, $(OBJS))
euai_test: $(TEST_OBJS) test/test_main.cpp
	$(CXX) $(CXXFLAGS) -DEUAI_TEST -o build/euai_test $(TEST_OBJS) test/test_main.cpp $(LDFLAGS)

# Python setup
python-install:
	pip3 install -r requirements.txt

python-data:
	python3 euai/python/data_prep.py

python-tokenizer:
	python3 euai/python/tokenizer_build.py

python-train:
	python3 euai/python/train.py

python-export:
	python3 euai/python/export.py

# Full pipeline
all-python: python-install python-data python-tokenizer python-train python-export
	@echo "Python pipeline complete. Model at python/euai.bin"

# Demo with synthetic data
demo: all-python
	@echo "Demo build complete. Run ./euai --model python/euai.bin --config config/"

# Quick run with test input
run: euai
	@echo "Testing: echo '2+2' | ./build/euai --config config/"
	echo "2+2" | ./build/euai --config config/ 2>/dev/null


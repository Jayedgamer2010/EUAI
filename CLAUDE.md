# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EUAI is a hybrid C++/Python language model implementation with the following components:

- **C++ Core**: Matrix operations, tokenizer, attention mechanism, model inference, and routing logic
- **Python Training**: PyTorch-based training pipeline with data preparation and model export
- **Router**: Query classification system that routes requests to specialized handlers (MATH, CACHE, SAFETY, NEURAL)
- **Quantization**: INT4 quantized model format for efficient inference (~35MB target)

## Repository Structure

```
euai/                    # Main C++ source code
├── src/
│   ├── core/            # Matrix math, tokenizer, model, attention
│   ├── router/          # Query classifier, safety screen, math engine, router
│   ├── inference/       # Inference engine and KV cache
│   └── main.cpp         # CLI entry point
├── python/              # Python training/export pipeline
│   ├── train.py         # PyTorch training loop
│   ├── export.py        # Model export with INT4 quantization
│   ├── data_prep.py     # Training data generation
│   ├── tokenizer_build.py  # BPE tokenizer training
│   ├── chat.py          # Python chat interface
│   └── data/            # Training data and tokenizer files
├── test/                # C++ test files
└── .claude/settings.local.json  # Claude Code permissions
```

## Common Development Tasks

### Environment Setup

```bash
# Install Python dependencies
pip install -r euai/requirements.txt

# Install system dependencies (Debian/Ubuntu)
sudo apt-get install g++ cmake nlohmann-json-dev libsqlite3-dev python3-dev

# For Termux (Android)
pkg install g++ cmake nlohmann-json-dev sqlite python
```

### Building the C++ Binary

```bash
# Navigate to euai directory
cd euai

# Build all object files
g++ -std=c++17 -O2 -c src/core/matrix.cpp
g++ -std=c++17 -O2 -c src/core/tokenizer.cpp
g++ -std=c++17 -O2 -c src/core/model.cpp
g++ -std=c++17 -O2 -c src/core/attention.cpp
g++ -std=c++17 -O2 -c src/router/router.cpp
g++ -std=c++17 -O2 -c src/router/classifier.cpp
g++ -std=c++17 -O2 -c src/router/safety.cpp
g++ -std=c++17 -O2 -c src/router/math_engine.cpp
g++ -std=c++17 -O2 -c src/inference/kvcache.cpp
g++ -std=c++17 -O2 -c src/inference/engine.cpp
g++ -std=c++17 -O2 -c src/main.cpp

# Link executable
g++ -std=c++17 -O2 -o euai *.o -lsqlite3

# Or use CMake (if CMakeLists.txt is configured)
mkdir -p build && cd build && cmake .. && make
```

### Running Tests

```bash
# Python validation (check PyTorch)
python -c "import torch; print('PyTorch available:', torch.__version__)"

# C++ core tests (compiles and runs tests)
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -I./src/core -I./test \
  src/core/matrix.cpp src/core/tokenizer.cpp src/core/model.cpp src/core/attention.cpp \
  test/test_main.cpp -o euai_core_test
./euai_core_test

# Or run main with EUAI_TEST defined
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -DEUAI_TEST \
  -I./src/core -I./src/router -I./src/inference \
  src/main.cpp src/core/matrix.cpp src/core/tokenizer.cpp src/core/model.cpp \
  src/core/attention.cpp src/router/router.cpp src/router/classifier.cpp \
  src/router/safety.cpp src/router/math_engine.cpp \
  src/inference/kvcache.cpp src/inference/engine.cpp \
  -lsqlite3 -o euai_test
./euai_test
```

### Training Pipeline

```bash
# 1. Prepare training data (generates python/data/train.txt)
python euai/python/data_prep.py

# 2. Build tokenizer (generates python/data/vocab.json, merges.json)
python euai/python/tokenizer_build.py

# 3. Train model (saves checkpoints to python/checkpoints/)
python euai/python/train.py

# Optional: run overfit test to verify model can learn
python euai/python/train.py --overfit-test

# 4. Export trained model to binary format (generates python/euai.bin)
python euai/python/export.py
```

### Running the Model

```bash
# Build the C++ binary first (see Building section)

# Run interactively
./euai --model python/euai.bin --config config/

# Arguments:
#   --model <path>    Path to quantized model binary (default: euai.bin)
#   --config <dir>    Config directory with vocab.json, merges.json, router_rules.json, cache.db (default: config/)
#   --max-tokens <n>  Maximum tokens to generate (passed to inference engine)
```

### Python Chat Interface

```bash
python euai/python/chat.py
```

## Architecture

### High-Level Flow

1. **Query Input** → Router
2. **Classification**: Determine query type (MATH, CODE, FACTUAL, CHAT, DANGEROUS, UNKNOWN)
3. **Safety Screen**: Block dangerous content
4. **Handler Selection**:
   - MATH → MathEngine (expression solver)
   - CACHE → SQLite cache lookup
   - NEURAL → InferenceEngine (language model)
   - SAFETY → Safety refusal messages
5. **Response**: Tagged with source and [END] marker

### C++ Core Components

- **Matrix**: Custom dense matrix class with operations (matmul, softmax, etc.)
- **Tokenizer**: BPE tokenizer using vocab.json and merges.json
- **Model**: Transformer with GQA attention, SwiGLU FFN, RMSNorm
- **KVCache**: KV cache for autoregressive generation
- **InferenceEngine**: Generation loop with top-k sampling
- **Router**: Multi-stage routing logic with SQLite caching

### Model Format (euai.bin)

Header:
- Magic: 0x45554149 ("EUAI")
- Config: dim, n_layers, n_heads, n_kv_groups, vocab_size, max_seq_len

Weights: Each tensor stored as:
- Shape (ndim, dims...)
- Scales (per-block absmax for INT4 quantization)
- Packed INT4 data (2 values per byte)

Target size: < 35MB

### Configuration Files

- `config/vocab.json`: Token vocabulary mapping
- `config/merges.json`: BPE merge rules
- `config/router_rules.json`: Classification patterns (safety_patterns, math_triggers)
- `config/cache.db`: SQLite cache for query results

## Code Conventions

- C++: Use `-std=c++17`, headers with `#pragma once`, nlohmann/json for JSON
- Python: Python 3, type hints optional, docstrings for main functions
- Error handling: C++ uses exceptions, check return values; Python uses print statements
- RAM monitoring: Check for 580MB threshold in training, output VmRSS in KB

## Performance Targets

- Training: 5000 steps, batch_size=2, grad_accumulation=16
- Inference: Target < 35MB model size, monitor RAM usage
- Quantization: INT4 per-block absmax, block_size=32

## Important Notes

- Model weights use INT4 quantization; dequantization happens at inference time
- RoPE ( Rotary Position Embedding ) is precomputed in cache during model loading
- KV cache uses per-layer, per-kv-group allocation
- Cache hit rate improves latency significantly
- Safety patterns use regex matching; dangerous queries are blocked before processing

## Testing Checklist

- [ ] Matrix multiplication accuracy (test: 2x2 matrices)
- [ ] Tokenizer encode/decode roundtrip
- [ ] Model loading from euai.bin
- [ ] Classification logic (math triggers, danger patterns)
- [ ] Math expression solver
- [ ] Cache database operations
- [ ] Full inference with temperature/top-k sampling

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python euai/python/export.py` after training |
| Tokenizer missing | Run `python euai/python/tokenizer_build.py` after data_prep |
| Build fails | Install nlohmann-json-dev and libsqlite3-dev |
| Out of memory during training | Reduce batch_size or use gradient accumulation |
| Quantized model exceeds 35MB | Adjust model dimensions (dim, n_layers) or block_size |

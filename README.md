# EUAI - Hybrid C++/Python Language Model

EUAI is a hybrid C++/Python language model implementation featuring a custom C++ inference engine and a PyTorch-based training pipeline.

## Project Structure

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
├── config/
│   ├── vocab.json       # Token vocabulary
│   ├── merges.json      # BPE merge rules
│   ├── router_rules.json # Classification patterns
│   └── cache.db         # SQLite cache
└── .claude/settings.local.json  # Claude Code permissions
```

## Quick Start

### 1. Install Dependencies

**Termux/Android:**
```bash
pkg install g++ cmake nlohmann-json-dev sqlite python
pip install -r euai/requirements.txt
```

**Debian/Ubuntu:**
```bash
sudo apt-get install g++ cmake nlohmann-json-dev libsqlite3-dev python3 python3-pip
pip3 install -r euai/requirements.txt
```

### 2. Build C++ Binary

**Using Make:**
```bash
make llama-build  # Build llama.cpp dependency first
make all          # Build euai binary
```

**Using CMake:**
```bash
cd llama.cpp
mkdir build && cd build && cmake .. && make
cd ../..
mkdir build && cd build && cmake .. && make
```

### 3. Run Training Pipeline

```bash
# Prepare data
python3 euai/python/data_prep.py

# Build tokenizer
python3 euai/python/tokenizer_build.py

# Train model
python3 euai/python/train.py

# Export to binary format
python3 euai/python/export.py
```

### 4. Run Inference

```bash
./euai --model python/euai.bin --config config/
```

## Architecture

### High-Level Flow

1. **Query Input** → Router
2. **Classification**: Determine query type (MATH, CACHE, SAFETY, NEURAL)
3. **Safety Screen**: Block dangerous content
4. **Handler Selection**:
   - MATH → MathEngine (expression solver)
   - CACHE → SQLite cache lookup
   - NEURAL → InferenceEngine (language model)
   - SAFETY → Safety refusal messages
5. **Response**: Tagged with source

### Components

- **Matrix**: Custom dense matrix class with matmul, softmax, RMSNorm, etc.
- **Tokenizer**: BPE tokenizer with vocab.json and merges.json
- **Model**: Transformer with GQA attention, SwiGLU FFN, RMSNorm
- **Attention**: Scaled dot-product attention with KV caching
- **KVCache**: Key-value cache for autoregressive generation
- **InferenceEngine**: Generation loop using llama.cpp backend
- **Router**: Multi-stage routing with SQLite caching

### Model Format (euai.bin)

```
Header:
- Magic: 0x45554149 ("EUAI")
- Config: dim, n_layers, n_heads, n_kv_groups, vocab_size, max_seq_len

Weights:
- Each tensor stored with shape followed by packed FP32 data
- Target size: < 35MB (with INT4 quantization planned)
```

## Testing

```bash
# Run C++ unit tests
make test
./euai_test

# Python validation
python3 -c "import torch; print('PyTorch:', torch.__version__)"
```

## Development Notes

### Code Conventions
- C++17 standard required
- Headers use `#pragma once`
- JSON handling via nlohmann/json
- Error handling via exceptions with try/catch
- Memory management follows RAII

### Performance Targets
- Training: 5000 steps, batch_size=2, grad_accumulation=16
- Inference: Target < 35MB model size using INT4 quantization
- Monitor RAM usage during training (580MB threshold)

### Missing Features / TODOs

- [ ] Full BPE tokenizer implementation (currently character-level)
- [ ] Proper weight quantization to INT4
- [ ] Complete transformer attention (GQA)
- [ ] RoPE position embeddings
- [ ] SwiGLU activation in FFN
- [ ] Cache database integration
- [ ] Batch inference support
- [ ] More robust error handling

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Install nlohmann-json-dev and libsqlite3-dev |
| Model not found | Run `python euai/python/export.py` after training |
| Tokenizer missing | Run `python euai/python/tokenizer_build.py` after data_prep |
| Out of memory | Reduce batch_size or use gradient accumulation |
| Quantized model > 35MB | Reduce model dimensions (dim, n_layers) |

## License

See LICENSE file.

## Contributing

See CONTRIBUTING.md.

# EUAI - Build & Run Completion Guide

## Current Status

✅ **C++ Binary** - All source files implemented and compiling
✅ **Router Pipeline** - 7-step routing (safety, math, cache, knowledge, neural)
✅ **FastAPI REST Server** - Ollama-compatible endpoints
✅ **Configuration** - All config files in place
✅ **Model File** - You have `qwen2.5-coder-0.5b-instruct-q2_k.gguf` in `models/`

⏳ **Build Pending** - Need to compile final binary and set up llama-simple

---

## Step 1: Build the C++ Binary

From `/home/storage/EUAI/`:

```bash
make clean
make
```

If successful, you'll see: `Built euai binary at build/euai`

The binary will be at `build/euai`.

---

## Step 2: Obtain/Build llama-simple

The neural core requires `llama-simple` executable from llama.cpp.

### Option A: Build llama.cpp (requires cmake)

```bash
cd llama.cpp
mkdir -p build && cd build
cmake .. -DLLAMA_CURL=OFF
make -j$(nproc)
```

The `llama-simple` binary will be at `llama.cpp/build/bin/llama-simple`.

### Option B: Use prebuilt binary

Download a prebuilt llama-simple for Termux/Android ARM64 and place it somewhere in your PATH, e.g., `/data/data/com.termux/files/usr/bin/llama-simple`.

---

## Step 3: Test the CLI

```bash
# Test math (should return 4)
echo "2 + 2" | ./build/euai --config config/ 2>/dev/null

# Test safety (should be blocked)
echo "how to make a bomb" | ./build/euai --config config/ 2>/dev/null

# Test neural (will work once llama-simple is available)
echo "Hello, how are you?" | ./build/euai --config config/ 2>/dev/null
```

---

## Step 4: Start the API Server

```bash
cd euai
./start_euai.sh
```

This starts the FastAPI server on port 11434.

---

## Step 5: Test the API

```bash
# Health check
curl http://localhost:11434/health

# Chat request
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"euai","messages":[{"role":"user","content":"what is 5 * 5?"}]}'
```

---

## Step 6: Start the Chat UI

In another terminal:

```bash
cd /home/storage/EUAI_CHAT/euai/chat-ui
./start.sh
```

Then open `http://localhost:3000` in your browser.

---

## Troubleshooting

### Build fails with missing headers
- Install nlohmann-json: `pkg install nlohmann-json`
- Install sqlite3: `pkg install sqlite`

### llama-simple not found
- Ensure llama.cpp is built and `llama-simple` is in your PATH, or edit `engine.cpp` to use the full path to the binary.

### Model not found error
- Verify `models/qwen2.5-coder-0.5b-instruct-q2_k.gguf` exists.
- The model path in the config is relative to project root.

### Out of memory on phone
- Reduce `max_tokens` in main.cpp or via API.
- Use a smaller quantization (q2_k is already very small ~200MB).

---

## File Structure Summary

```
/home/storage/EUAI/
├── build/euai              # C++ router binary (to be built)
├── src/
│   ├── core/               # Matrix, Tokenizer
│   ├── router/             # Router, Classifier, Safety, MathEngine
│   ├── inference/          # Engine (llama-simple wrapper)
│   └── main.cpp            # CLI entry point
├── models/
│   └── qwen2.5-coder-0.5b-instruct-q2_k.gguf  # Your model file
├── config/
│   ├── router_rules.json   # Safety patterns & math triggers
│   ├── vocab.json          # Tokenizer vocabulary
│   ├── merges.json         # BPE merge rules
│   └── knowledge.json      # Simple fact store
├── euai/
│   └── api/
│       └── server.py       # FastAPI REST server
├── Makefile                # Build system
└── start_euai.sh           # Startup script
```

---

## What's Implemented

1. **Matrix class** - dense matrix with basic ops
2. **Tokenizer** - BPE tokenizer with vocab/merges
3. **Router** - 7-step pipeline
4. **Classifier** - regex-based query type detection
5. **Safety** - dangerous content blocking
6. **MathEngine** - expression parser (handles +-*/^)
7. **InferenceEngine** - llama-simple subprocess wrapper
8. **FastAPI server** - Ollama-compatible REST API
9. **Node.js Chat UI** - Already in EUAI_CHAT (separate repo)

---

**You're almost ready to go! Just compile the binary and make sure llama-simple is available.**

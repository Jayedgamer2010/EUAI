# EUAI — Complete Project Documentation
> Efficient Updated Artificial Intelligence  
> Built by Jayed Sheikh | Started 2025

---

## Table of Contents
1. [What is EUAI](#what-is-euai)
2. [Architecture Overview](#architecture-overview)
3. [How It Works](#how-it-works)
4. [File & Folder Structure](#file--folder-structure)
5. [The 7-Step Router Pipeline](#the-7-step-router-pipeline)
6. [Neural Core](#neural-core)
7. [Training System](#training-system)
8. [Inference Engine](#inference-engine)
9. [REST API](#rest-api)
10. [Chat UI](#chat-ui)
11. [Authentication & Database](#authentication--database)
12. [Deployment](#deployment)
13. [Hardware Requirements](#hardware-requirements)
14. [What's Done](#whats-done)
15. [What's In Progress](#whats-in-progress)
16. [What's Planned](#whats-planned)
17. [Known Issues](#known-issues)
18. [How to Run](#how-to-run)

---

## What is EUAI

EUAI is a **custom AI system built from scratch** — not a fine-tuned version of GPT, Llama, or any existing model infrastructure. It is a hybrid AI that combines:

- A **7-step routing pipeline** that handles 90% of queries without touching a neural model
- A **C++ inference engine** (70% C++) that runs on CPU with no GPU
- A **Qwen2.5 0.5B GGUF model** as the neural core
- A **FastAPI REST server** compatible with the Ollama API format
- A **Node.js + Supabase chat UI** with authentication and persistent history

**The core philosophy:** Route queries away from the heavy neural model whenever possible. Math goes to a math engine. Dangerous queries get blocked. Repeated questions come from cache. Only complex, creative queries touch the neural model.

**Built and runs on:** Snapdragon 430 Android phone via Termux. No GPU. ~300MB RAM at runtime.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           C++ Router Pipeline           │
│  Step 1: Query Classifier               │
│  Step 2: Safety Screen                  │
│  Step 3: Pattern Cache (SQLite)         │
│  Step 4: Math Engine                    │
│  Step 5: Knowledge Store                │
│  Step 6: Neural Core (last resort)      │
│  Step 7: Response Merge + Source Tag    │
└─────────────────────────────────────────┘
    │
    ▼
[SOURCE: MATH] / [SOURCE: CACHE] / [SOURCE: NEURAL] / [SOURCE: SAFETY]
```

**Language split:**
- **70% C++** — all inference code, router, math engine, safety, KV cache, tokenizer
- **30% Python** — training pipeline (runs once), FastAPI REST server, thin CLI wrapper

---

## How It Works

### Full Data Flow

```
1. User types message in Chat UI (browser)
        ↓
2. Chat UI (Node.js, port 3000) proxies request
        ↓
3. EUAI REST API (FastAPI, port 11434)
        ↓
4. router.cpp — 7-step pipeline
        ↓
5a. Math query   → math_engine.cpp → exact answer
5b. Danger query → safety.cpp → refusal message
5c. Cached query → SQLite cache → stored answer
5d. Complex query → engine.cpp → llama-simple subprocess
        ↓
6. llama-simple loads qwen2.5-0.5b.gguf, generates tokens
        ↓
7. Response streams back token by token via SSE
        ↓
8. Chat UI renders tokens in real time
        ↓
9. Message saved to Supabase database
```

### Training Flow (one time)

```
data_prep.py → train.txt (521MB, 491k documents)
        ↓
tokenizer_build.py → vocab.json + merges.json (8192 BPE tokens)
        ↓
train.py (PyTorch, GPU on Colab/Modal) → checkpoints/best.pt
        ↓
export.py → euai.bin (INT4 quantized, ~28MB)
        ↓
C++ engine loads euai.bin via mmap()
```

> **Note:** Current deployment uses Qwen2.5 0.5B GGUF directly via llama-simple subprocess instead of the custom-trained model. Custom training is paused pending GPU access.

---

## File & Folder Structure

```
/home/storage/EUAI/
├── euai/                          ← Main EUAI project
│   ├── src/
│   │   ├── core/
│   │   │   ├── matrix.h/cpp       ← Matrix math ops (matmul, softmax, silu, rmsnorm)
│   │   │   ├── tokenizer.h/cpp    ← C++ BPE tokenizer, loads vocab.json
│   │   │   ├── model.h/cpp        ← Load euai.bin via mmap, forward pass
│   │   │   └── attention.h/cpp    ← GQA attention + RoPE + KV cache
│   │   ├── router/
│   │   │   ├── router.h/cpp       ← 7-step pipeline orchestrator
│   │   │   ├── classifier.h/cpp   ← Query type detection via regex
│   │   │   ├── safety.h/cpp       ← Block dangerous queries
│   │   │   └── math_engine.h/cpp  ← Recursive descent math parser
│   │   ├── inference/
│   │   │   ├── engine.h/cpp       ← Inference engine (calls llama-simple subprocess)
│   │   │   └── kvcache.h/cpp      ← KV cache management
│   │   └── main.cpp               ← Entry point, arg parsing, stdin loop
│   ├── api/
│   │   ├── server.py              ← FastAPI REST server (Ollama-compatible)
│   │   ├── runner.py              ← Calls ./build/euai binary, parses output
│   │   └── models.py              ← Pydantic request/response schemas
│   ├── python/
│   │   ├── data_prep.py           ← Download + clean training data
│   │   ├── tokenizer_build.py     ← Build BPE vocab from train.txt
│   │   ├── train.py               ← PyTorch training loop
│   │   ├── export.py              ← checkpoint → INT4 → euai.bin
│   │   └── chat.py                ← Thin CLI wrapper
│   ├── config/
│   │   ├── model.json             ← Architecture hyperparams
│   │   ├── router_rules.json      ← Safety patterns + math triggers
│   │   └── euai.env               ← Paths config (loaded by runner.py)
│   ├── data/
│   │   └── train.txt              ← 521MB training data (491k documents)
│   ├── models/
│   │   └── qwen2.5-0.5b.gguf     ← Qwen2.5 0.5B INT4 model (~350MB)
│   ├── build/
│   │   └── euai                   ← Compiled C++ binary
│   ├── start_euai.sh              ← Start API + Cloudflare tunnel
│   └── stop_euai.sh               ← Stop everything
│
├── llama.cpp/                     ← llama.cpp (built for CPU, Termux)
│   └── build/
│       ├── src/libllama.a
│       ├── common/libcommon.a
│       ├── ggml/src/libggml*.a
│       └── bin/llama-simple       ← Used by engine.cpp
│
└── EUAI_CHAT/
    └── euai/
        └── chat-ui/               ← Chat web UI
            ├── server.js          ← Express server, proxies to EUAI API
            ├── public/
            │   ├── index.html     ← Main chat interface
            │   ├── auth.html      ← Login/signup page
            │   ├── app.js         ← Frontend logic (vanilla JS)
            │   └── style.css      ← Dark terminal theme
            ├── .env               ← Supabase credentials + port
            ├── start.sh
            └── stop.sh
```

---

## The 7-Step Router Pipeline

Every query passes through these steps in order. Steps 1-5 cost near zero RAM and respond in <50ms.

| Step | Name | What it does | Speed | RAM |
|------|------|-------------|-------|-----|
| 1 | Query Classifier | Detect type: MATH, CODE, FACTUAL, CHAT, DANGEROUS | ~1ms | 0 |
| 2 | Safety Screen | Block harmful/dangerous queries, return refusal | ~2ms | 0 |
| 3 | Pattern Cache | Check SQLite for repeated questions, return cached answer | ~5ms | ~10MB |
| 4 | Math Engine | Recursive descent parser for arithmetic/sqrt/pow | ~10ms | 0 |
| 5 | Knowledge Store | Keyword search in knowledge.json | ~20ms | ~5MB |
| 6 | Neural Core | llama-simple generates response (last resort only) | ~5-30s | ~350MB |
| 7 | Response Merge | Tag answer with source, return to user | ~1ms | 0 |

**Source tags:**
- `[SOURCE: MATH]` — answered by math engine
- `[SOURCE: CACHE]` — answered from SQLite cache
- `[SOURCE: SAFETY]` — blocked by safety screen
- `[SOURCE: NEURAL]` — answered by Qwen2.5 model

---

## Neural Core

### Current Model: Qwen2.5 0.5B Instruct (GGUF)

| Property | Value |
|----------|-------|
| Parameters | 500M |
| Quantization | Q4_0 (INT4) |
| File size | ~350MB |
| Context window | 32,768 tokens |
| Format | GGUF |
| Source | HuggingFace: Qwen/Qwen2.5-0.5B-Instruct-GGUF |

**Chat template (Qwen2.5 format):**
```
<|im_start|>system
You are EUAI. Answer concisely.
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
```

**EOS tokens:** 151643 (`<|endoftext|>`), 151645 (`<|im_end|>`)

**Sampler chain:**
- Top-K: 40
- Top-P: 0.95
- Temperature: 0.8
- Repeat penalty: 1.15, freq: 0.1

### Custom EUAI Model (planned/paused)

Architecture designed from scratch:
- 15M params (prototype) → 1.39B params (full)
- GQA (Grouped Query Attention) — smaller KV cache
- SwiGLU activation — better quality
- RoPE positional encoding
- RMSNorm (faster than LayerNorm)
- Pre-norm architecture (stable from-scratch training)
- Weight tying (embed ↔ lm_head)

**Prototype specs (15M):**
- dim=384, n_layers=8, n_heads=6, n_kv_groups=2
- vocab=4096, max_seq_len=256
- Trains in ~600MB RAM on CPU

**Full model specs (1.39B):**
- dim=2048, n_layers=24, n_heads=16, n_kv_groups=4
- vocab=32000, max_seq_len=4096
- Trains on A40 GPU (~$25 one time on RunPod)

---

## Training System

### Training Data (521MB, 491,564 documents)

| Dataset | Size | What it teaches |
|---------|------|----------------|
| TinyStories (150k) | ~60MB | Clean sentence structure |
| Wikipedia Simple (full) | ~200MB | Factual knowledge |
| OpenHermes (75k) | ~120MB | Q&A, instruction following |
| Open Platypus (full) | ~50MB | Step-by-step reasoning |

### Training Config

```python
batch_size = 2
grad_accumulation = 16  # effective batch = 32
lr = 3e-4
warmup_steps = 200
max_steps = 5000
grad_clip = 1.0
optimizer = AdamW(weight_decay=0.1)
scheduler = cosine with warmup
```

### RAM Budget During Training

| Component | RAM |
|-----------|-----|
| PyTorch + Python | ~150MB |
| Model weights (FP32) | ~56MB |
| Adam optimizer states | ~112MB |
| Gradient buffers | ~56MB |
| Activations | ~80MB |
| Data loader | ~30MB |
| **Total peak** | **~484-560MB** |

---

## Inference Engine

### How engine.cpp Works

```
engine.cpp receives prompt string
    ↓
Writes Qwen2.5 chat format to /tmp/euai_in.txt
    ↓
Calls: llama-simple -m models/qwen2.5-0.5b.gguf -n 2048 "$(cat /tmp/euai_in.txt)"
    ↓
Reads output from /tmp/euai_out.txt
    ↓
Parses everything after <|im_start|>assistant line
    ↓
Returns clean response string
```

### RAM at Runtime (C++ inference)

| Component | RAM |
|-----------|-----|
| C++ binary | ~5MB |
| llama-simple process | ~400MB (includes model) |
| EUAI router + cache | ~20MB |
| FastAPI server (Python) | ~80MB |
| **Total** | **~505MB** |

---

## REST API

**Base URL:** `http://localhost:11434` (or `https://euaiweb.darkaether.store` via Cloudflare)

**Ollama-compatible endpoints:**

### GET /health
```json
{
  "status": "ok",
  "model": "qwen2.5-0.5b",
  "model_loaded": true,
  "binary_ready": true,
  "version": "1.0"
}
```

### GET /api/tags
```json
{
  "models": [{"name": "euai", "modified_at": "...", "size": 350000000}]
}
```

### POST /api/chat
```json
// Request
{"model": "euai", "messages": [{"role": "user", "content": "tell me a joke"}], "stream": false}

// Response
{"model": "euai", "message": {"role": "assistant", "content": "Why did..."}, "done": true}
```

### POST /api/chat/stream
Same as above but returns SSE stream:
```
data: {"token": "W", "done": false}
data: {"token": "h", "done": false}
data: {"token": "y", "done": false}
...
data: {"token": "", "done": true}
```

### POST /api/generate
```json
// Request
{"model": "euai", "prompt": "tell me a joke"}

// Response
{"model": "euai", "response": "Why did...", "done": true}
```

---

## Chat UI

**Stack:** Node.js + Express + Vanilla JS (no React, no build step)  
**Port:** 3000  
**Public:** `https://euaiweb.darkaether.store` (via Cloudflare Tunnel)

### Features
- Dark terminal theme (#0a0a0f background)
- Left sidebar with chat history grouped by Today/Yesterday/This Week
- Real-time streaming — tokens appear as they generate
- Source badges: [MATH] green, [NEURAL] cyan, [CACHE] yellow, [SAFETY] red
- Mobile responsive — sidebar collapses on phone
- Auto-resize textarea input
- Enter to send, Shift+Enter for newline

### Server Routes (server.js)
| Route | Method | What it does |
|-------|--------|-------------|
| `/` | GET | Serve index.html |
| `/health` | GET | Check EUAI API status |
| `/config` | GET | Return Supabase config to frontend |
| `/chat` | POST | Forward to EUAI API, return response |
| `/chat/stream` | POST | Forward SSE stream from EUAI API |
| `/conversations` | POST | Create conversation in Supabase |
| `/conversations/:uid` | GET | Get user's conversation list |
| `/messages/:cid` | GET | Get messages for conversation |
| `/messages/save` | POST | Save message to Supabase |

---

## Authentication & Database

**Provider:** Supabase (free tier)

### Database Schema

**conversations table:**
```sql
id           uuid primary key default gen_random_uuid()
user_id      uuid references auth.users(id) on delete cascade
title        text default 'New Chat'
created_at   timestamptz default now()
updated_at   timestamptz default now()
```

**messages table:**
```sql
id                uuid primary key default gen_random_uuid()
conversation_id   uuid references conversations(id) on delete cascade
role              text  -- 'user' or 'assistant'
content           text
source            text default 'NEURAL'
created_at        timestamptz default now()
```

**Row Level Security:** Users can only access their own conversations and messages.

### Auth Flow
1. User opens `euaiweb.darkaether.store`
2. Not logged in → redirect to `/auth.html`
3. Login/signup via Supabase Auth (email + password)
4. Session stored in localStorage
5. On chat: messages auto-saved to Supabase
6. History persists across sessions and devices

---

## Deployment

### Current Setup

```
Phone (Snapdragon 430, 2.77GB RAM, Termux)
    ├── EUAI API (port 11434) ← uvicorn + FastAPI
    ├── Chat UI (port 3000)   ← Node.js + Express
    └── Cloudflare Tunnel     ← euaiweb.darkaether.store → localhost:11434
```

### Start Everything
```bash
# Terminal 1 — EUAI API + Cloudflare tunnel
/home/storage/EUAI/euai/start_euai.sh

# Terminal 2 — Chat UI
/home/storage/EUAI_CHAT/euai/chat-ui/start.sh
```

### Stop Everything
```bash
/home/storage/EUAI/euai/stop_euai.sh
/home/storage/EUAI_CHAT/euai/chat-ui/stop.sh
```

### Compile C++ (after any code changes)
```bash
cd /home/storage/EUAI/euai
g++ -std=c++17 -O2 -fopenmp \
  -I./src/inference -I./src/router \
  -I/home/storage/EUAI/llama.cpp/include \
  -I/home/storage/EUAI/llama.cpp/ggml/include \
  -I/data/data/com.termux/files/usr/include \
  -I/data/data/com.termux/files/home/storage/EUAI/llama.cpp/vendor \
  src/router/classifier.cpp src/router/safety.cpp \
  src/router/math_engine.cpp src/router/router.cpp \
  src/inference/engine.cpp src/main.cpp \
  /home/storage/EUAI/llama.cpp/build/src/libllama.a \
  /home/storage/EUAI/llama.cpp/build/common/libcommon.a \
  /home/storage/EUAI/llama.cpp/build/ggml/src/libggml.a \
  /home/storage/EUAI/llama.cpp/build/ggml/src/libggml-base.a \
  /home/storage/EUAI/llama.cpp/build/ggml/src/libggml-cpu.a \
  -lsqlite3 -ldl -lm -lpthread -lomp \
  -o build/euai 2>&1 && echo "COMPILE OK"
```

### Required env vars (OMP fixes for Snapdragon 430)
```bash
export OMP_NUM_THREADS=1
export KMP_AFFINITY=disabled
```

---

## Hardware Requirements

### To Run (current setup)

| Component | Minimum | Current Phone |
|-----------|---------|--------------|
| RAM | 512MB | 2.77GB ✅ |
| Storage | 800MB | 25GB ✅ |
| CPU | Any ARM | Snapdragon 430 ✅ |
| GPU | None needed | None ✅ |
| OS | Linux/Android | Android ✅ |

### To Train Custom Model

| Hardware | Spec | Cost |
|----------|------|------|
| GPU (cloud) | T4 16GB | ~$0.35/hr |
| Training time | ~1-2 hrs (15M) | ~$0.50 total |
| Full model (1.39B) | A40 48GB | ~$25 one time |

---

## What's Done

- [x] C++ transformer architecture (GQA, SwiGLU, RoPE, RMSNorm)
- [x] C++ BPE tokenizer
- [x] 7-step hybrid routing pipeline
- [x] Math engine (recursive descent parser)
- [x] Safety screen (pattern matching)
- [x] SQLite pattern cache
- [x] llama.cpp integration via subprocess
- [x] Qwen2.5 0.5B GGUF inference working
- [x] FastAPI REST server (Ollama-compatible)
- [x] Token-by-token streaming via SSE
- [x] Node.js + Express chat UI
- [x] Real-time streaming in browser
- [x] Supabase auth (login/signup)
- [x] Persistent chat history in Supabase
- [x] Cloudflare Tunnel (euaiweb.darkaether.store)
- [x] Training data pipeline (521MB, 491k docs)
- [x] BPE tokenizer builder (8192 vocab)
- [x] 7 critical fixes (CORS, health endpoint, error handling, etc.)
- [x] Mobile responsive UI
- [x] Source tags on responses

---

## What's In Progress

- [ ] Chat UI full redesign (Claude.ai level quality)
- [ ] Fix Next.js → revert to Express (Next.js too heavy for phone)
- [ ] Recover files deleted during git push incident
- [ ] Conversation history display bug fixes

---

## What's Planned

### Short Term (next sessions)
- [ ] UI redesign Phase 3 — Claude-level professional interface
- [ ] Multi-turn conversation memory (pass history to model)
- [ ] Code syntax highlighting in chat
- [ ] Markdown rendering in responses
- [ ] Dark/light mode toggle

### Medium Term
- [ ] Train custom 15M prototype model on Colab/Modal
- [ ] Deploy chat UI to Vercel/Netlify
- [ ] FastAPI server on cheap VPS ($3/month)
- [ ] Mobile Android app wrapping the web UI
- [ ] Knowledge base — load custom facts

### Long Term
- [ ] Train full 1.39B custom model on A40 (~$25)
- [ ] Replace Qwen2.5 with custom EUAI weights
- [ ] Tool use (web search, code execution)
- [ ] RAG system (connect to documents)
- [ ] API keys + rate limiting for sharing

---

## Known Issues

| Issue | Status | Workaround |
|-------|--------|-----------|
| Long responses freeze phone (CPU heat) | Known | Keep max_tokens reasonable |
| OMP assertion crash without env vars | Fixed | Set OMP_NUM_THREADS=1 KMP_AFFINITY=disabled |
| Next.js too slow to compile on SD430 | Fixed | Reverted to Express + vanilla JS |
| Supabase MCP DNS errors in Termux | Known | Create tables manually via SQL Editor |
| Model repeats response sometimes | Partially fixed | Penalty sampler + repeat detection |
| llama-simple not llama-cli available | Known | llama-simple works fine as replacement |

---

## How to Run

### Quick Start
```bash
# Start EUAI (API + Cloudflare tunnel)
/home/storage/EUAI/euai/start_euai.sh

# Start Chat UI (new Termux session)
/home/storage/EUAI_CHAT/euai/chat-ui/start.sh
```

### Access
- **Local:** `http://192.168.0.100:3000`
- **Public:** `https://euaiweb.darkaether.store`

### Test API directly
```bash
# Health check
curl http://localhost:11434/health

# Math (uses router, not neural)
echo "what is 25 * 25" | ./build/euai --config config/ --model models/qwen2.5-0.5b.gguf 2>/dev/null

# Neural
echo "tell me a joke" | ./build/euai --config config/ --model models/qwen2.5-0.5b.gguf 2>/dev/null

# Safety block
echo "how to hack a website" | ./build/euai --config config/ --model models/qwen2.5-0.5b.gguf 2>/dev/null
```

---

## Project Stats

| Metric | Value |
|--------|-------|
| Total development time | ~8+ hours |
| Lines of C++ code | ~2000+ |
| Lines of Python code | ~800+ |
| Lines of JS/HTML/CSS | ~1500+ |
| Training data | 521MB / 491,564 docs |
| Model size | ~350MB (Qwen2.5 Q4_0) |
| Runtime RAM | ~505MB |
| Response time (math) | <50ms |
| Response time (neural) | 5-30 seconds |
| Built on | Snapdragon 430 Android phone |

---

*EUAI — proving that you don't need a datacenter to build a real AI system from scratch.*  
*Built by Jayed Sheikh, Bangladesh, Class 10, age 16.*

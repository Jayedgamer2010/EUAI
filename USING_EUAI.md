# Using EUAI - CLI & API Guide

## Quick Start

### Start EUAI Server

```bash
./euai/start_euai.sh
```

The server starts on **port 11434** (Ollama-compatible API).

**Output:**
```
=== Starting EUAI ===
[INFO] Starting FastAPI server on port 11434...
[INFO] API server started with PID <pid>

EUAI is running!
  Local API: http://localhost:11434
  Health:    http://localhost:11434/health

To stop: /home/storage/EUAI/euai/stop_euai.sh
```

### Stop EUAI Server

```bash
./euai/stop_euai.sh
```

## API Usage (Ollama-compatible)

EUAI provides a REST API compatible with Ollama. Use any Ollama client or curl.

### Health Check

```bash
curl http://localhost:11434/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "euai",
  "model_loaded": true,
  "binary_ready": true,
  "version": "1.0"
}
```

### List Models

```bash
curl http://localhost:11434/api/tags
```

**Response:**
```json
{
  "models": [
    {
      "name": "euai",
      "modified_at": "2025-03-25T00:00:00Z",
      "size": 350000000
    }
  ]
}
```

### Chat (Non-streaming)

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "euai",
    "messages": [
      {"role": "user", "content": "what is 15 * 15"}
    ]
  }'
```

**Response:**
```json
{
  "model": "euai",
  "message": {
    "role": "assistant",
    "content": "[SOURCE: MATH] Result: 225.000000"
  },
  "done": true
}
```

### Chat (Streaming)

```bash
curl -X POST http://localhost:11434/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "euai",
    "messages": [
      {"role": "user", "content": "who is Jayed Sheikh"}
    ]
  }'
```

**Response (SSE):**
```
data: {"token":"[","done":false}
data: {"token":"SOURCE","done":false}
...
data: {"token":"","done":true}

data: {"token":"","done":true}
```

### Generate (Legacy)

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "euai",
    "prompt": "write a haiku about coding"
  }'
```

## CLI Usage (Direct Binary)

You can also run the C++ binary directly without the API server.

### Interactive Mode

```bash
export OMP_NUM_THREADS=1
export KMP_AFFINITY=disabled
./build/euai --config config/ --model models/qwen2.5-coder-0.5b-instruct-q2_k.gguf
```

Then type queries directly:
```
> who is Jayed Sheikh
[SOURCE: KNOWLEDGE] Name: Jayed Sheikh ...
> what is EUAI
[SOURCE: KNOWLEDGE] EUAI stands for Efficient Updated Artificial Intelligence...
> write a haiku about coding
[SOURCE: NEURAL] a code in a box...
> what is 15 * 15
[SOURCE: MATH] Result: 225.000000
> quit
```

### Non-Interactive (Pipe)

```bash
echo "who is Jayed Sheikh" | ./build/euai --config config/ --model models/qwen2.5-coder-0.5b-instruct-q2_k.gguf
```

**Output:**
```
[SOURCE: KNOWLEDGE] Name: Jayed Sheikh
Age: 16
Location: Bangladesh, Class 10
...
```

### Command-Line Options

```
--model <path>       Path to GGUF model (default: models/qwen2.5-coder-0.5b-instruct-q2_k.gguf)
--config <dir>       Config directory (default: config/)
--max-tokens <n>     Maximum tokens to generate (default: 200)
--stats              Print statistics on exit
-h, --help           Show help message
```

## Features in Action

### 1. Knowledge Folder

EUAI automatically loads content from the `knowledge/` directory.

**Test:**
```bash
echo "who is Jayed" | ./build/euai --config config/ --model models/qwen2.5-coder-0.5b-instruct-q2_k.gguf
```

**Output:** Returns info from `knowledge/about_me.txt` with `[SOURCE: KNOWLEDGE]`

### 2. System Prompt Personality

The system prompt is loaded from `config/system_prompt.txt`. Edit this file to change EUAI's personality without recompiling.

**Current prompt:**
```
You are EUAI, a personal AI assistant built by Jayed Sheikh from Bangladesh.
You are helpful, direct, and concise in your responses.
You are running on a Snapdragon 430 Android phone using Termux.
You have expertise in programming, game development, AI, and online business.
Always respond in the same language the user writes in.
Keep responses short and clear unless the user asks for detail.
Never repeat yourself. Never repeat the same sentence twice in one response.
```

### 3. Routing Pipeline

EUAI uses a 7-step hybrid routing:

1. **Safety Screen** - Blocks dangerous queries (outputs `[SOURCE: SAFETY]`)
2. **Math Engine** - Solves math expressions (outputs `[SOURCE: MATH]`)
3. **Cache Lookup** - Returns cached answers (outputs `[SOURCE: CACHE]`)
4. **Knowledge Store** - Searches custom knowledge files (outputs `[SOURCE: KNOWLEDGE]`)
5. **Neural Model** - Qwen2.5 0.5B for general queries (outputs `[SOURCE: NEURAL]`)
6. **Fallback** - Unknown queries get default response

**Test routing:**

```bash
# Safety (blocked)
echo "how do I hack" | ./build/euai --config config/ --model models/qwen2.5-coder-0.5b-instruct-q2_k.gguf

# Math
echo "225 / 15" | ./build/euai --config config/ --model models/qwen2.5-coder-0.5b-instruct-q2_k.gguf

# Knowledge
echo "what is EUAI" | ./build/euai --config config/ --model models/qwen2.5-coder-0.5b-instruct-q2_k.gguf

# Neural (creative)
echo "write python hello world" | ./build/euai --config config/ --model models/qwen2.5-coder-0.5b-instruct-q2_k.gguf
```

## Configuration Files

| File | Purpose |
|------|---------|
| `config/system_prompt.txt` | System prompt / personality |
| `config/router_rules.json` | Classification rules, cache TTL, safety patterns |
| `config/vocab.json` | Tokenizer vocabulary |
| `config/merges.json` | BPE merge rules |
| `config/cache.db` | SQLite query cache (auto-created) |
| `knowledge/*.txt` | Custom knowledge base (plain text, paragraphs separated by blank lines) |
| `knowledge/*.json` | Knowledge in JSON format |

### Adding Knowledge Files

**Plain text (.txt):**
```
Topic: History
EUAI was created in 2025 by Jayed Sheikh.
It runs on Android using Termux.

Topic: Features
EUAI supports:
- Knowledge retrieval
- Math solving
- Neural generation
```

**JSON (.json):**
```json
[
  {
    "content": "EUAI is a hybrid C++/Python language model."
  },
  {
    "content": "The Snapdragon 430 has 8 cores and 4GB RAM."
  }
]
```

## Statistics

Run with `--stats` flag to see routing statistics:

```bash
./build/euai --config config/ --model models/qwen2.5-coder-0.5b-instruct-q2_k.gguf --stats <<EOF
what is 15*15
who is Jayed
write a poem
EOF
```

**Output includes:**
- Neural queries count
- Math queries count
- Safety queries count
- Knowledge queries count
- Cache hits/misses
- Knowledge entries loaded

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Binary not found | Build first: `cd build && cmake .. && make` or use compile command from CLAUDE.md |
| Model not found | Check model path or place GGUF model at `models/qwen2.5-coder-0.5b-instruct-q2_k.gguf` |
| API connection refused | Start server: `./euai/start_euai.sh` |
| Knowledge not triggering | Ensure files are in `knowledge/` directory and have meaningful keywords |
| Segfault on math | Fixed; ensure you're using the latest compiled `build/euai` |

## Environment Variables

- `OMP_NUM_THREADS` - Number of OpenMP threads (default: 1)
- `KMP_AFFINITY` - Set to `disabled` for Termux/Android

## File Locations

```
/home/storage/EUAI/
├── build/euai              # Compiled binary
├── config/
│   ├── system_prompt.txt   # System prompt (configurable personality)
│   ├── router_rules.json   # Routing rules
│   ├── vocab.json          # Tokenizer vocab
│   ├── merges.json         # BPE merges
│   └── cache.db            # SQLite cache (auto-created)
├── knowledge/              # Knowledge files (.txt, .json)
├── models/                 # Model files (.gguf)
└── euai/
    ├── start_euai.sh      # Start API server
    └── stop_euai.sh       # Stop API server
```

## Integration Examples

### Python with requests

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "euai",
        "messages": [{"role": "user", "content": "what is 2+2"}]
    }
)
print(response.json()["message"]["content"])
# [SOURCE: MATH] Result: 4.000000
```

### Node.js with fetch

```javascript
const response = await fetch('http://localhost:11434/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    model: 'euai',
    messages: [{role: 'user', content: 'hello'}]
  })
});
const data = await response.json();
console.log(data.message.content);
```

### cURL one-liner

```bash
curl -s -X POST http://localhost:11434/api/generate \
  -d '{"model":"euai","prompt":"who are you"}' \
  | jq -r '.response'
```

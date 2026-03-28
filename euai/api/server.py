#!/usr/bin/env python3
"""
FastAPI REST Server - Ollama-compatible API for EUAI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import *
from api.runner import Runner

app = FastAPI(title="EUAI API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
BINARY_PATH = Path(__file__).parent.parent.parent / "build" / "euai"
CONFIG_DIR = Path(__file__).parent.parent / "config"

# Initialize runner
try:
    runner = Runner(BINARY_PATH, CONFIG_DIR)
    print(f"[API] Runner initialized with binary: {BINARY_PATH}")
    print(f"[API] Model path: {runner.model_path}")
except Exception as e:
    print(f"[API] Failed to initialize runner: {e}")
    runner = None

@app.get("/health", response_model=HealthResponse)
async def health():
    if runner:
        return runner.health_check()
    return HealthResponse(
        status="error",
        model="euai",
        model_loaded=False,
        binary_ready=False
    )

@app.get("/api/tags", response_model=TagsResponse)
async def tags():
    return TagsResponse(
        models=[{
            "name": "euai",
            "modified_at": "2025-03-25T00:00:00Z",
            "size": 350000000
        }]
    )

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""
    if not runner:
        raise HTTPException(status_code=503, detail="Runner not initialized")

    # Build prompt from messages
    prompt = ""
    if request.messages:
        last_msg = request.messages[-1]
        prompt = last_msg.content

    source, response = runner.generate(prompt, max_tokens=200)

    # Include source tag in response for client to display
    full_response = f"[SOURCE: {source}] {response}"

    return ChatResponse(
        model=request.model,
        message=Message(role="assistant", content=full_response),
        done=True
    )

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    if not runner:
        raise HTTPException(status_code=503, detail="Runner not initialized")

    prompt = ""
    if request.messages:
        last_msg = request.messages[-1]
        prompt = last_msg.content

    source, response = runner.generate(prompt, max_tokens=200)
    # Include source tag at the beginning of the streamed response
    full_response = f"[SOURCE: {source}] {response}"

    async def stream_gen():
        for char in full_response:
            yield f"data: {json.dumps({'token': char, 'done': False})}\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(stream_gen(), media_type="text/event-stream")

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate endpoint (legacy)"""
    if not runner:
        raise HTTPException(status_code=503, detail="Runner not initialized")

    source, response = runner.generate(request.prompt, max_tokens=200)
    full_response = f"[SOURCE: {source}] {response}"

    return GenerateResponse(
        model=request.model,
        response=full_response,
        done=True
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)

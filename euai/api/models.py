from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "euai"
    messages: List[Message]
    stream: bool = False
    options: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    model: str
    message: Message
    done: bool

class StreamChunk(BaseModel):
    token: str
    done: bool

class GenerateRequest(BaseModel):
    model: str = "euai"
    prompt: str
    stream: bool = False

class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool

class TagsResponse(BaseModel):
    models: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    binary_ready: bool
    version: str = "1.0"

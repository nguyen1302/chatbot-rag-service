from pydantic import BaseModel
from typing import List

class ChatMessage(BaseModel):
    role: str      # "user", "system", "assistant"
    content: str

class RAGRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False

class RAGResponse(BaseModel):
    prompt: str
    context_chunks: List[str] = []

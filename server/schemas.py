"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    cached: bool = False


class DocumentInfo(BaseModel):
    name: str
    source_path: str
    chunks: int
    size_bytes: int = 0


class SettingsResponse(BaseModel):
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"
    top_k: int = 5
    rerank_enabled: bool = True
    stream_enabled: bool = True
    chunk_size: int = 1024
    chunk_overlap: int = 200


class SettingsUpdate(BaseModel):
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    top_k: int | None = None
    rerank_enabled: bool | None = None
    stream_enabled: bool | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class IngestResponse(BaseModel):
    indexed: int
    total_chunks: int
    errors: list[str] = []


class DeleteResponse(BaseModel):
    status: str
    chunks_removed: int = 0

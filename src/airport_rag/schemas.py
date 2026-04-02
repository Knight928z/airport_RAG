from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    index: int
    source: str
    page: Optional[int] = None
    chunk_id: str
    snippet: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2, description="机场业务问题")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    confidence_note: str


class IngestRequest(BaseModel):
    input_path: str = Field(description="文档目录或文件路径")


class IngestResponse(BaseModel):
    indexed_chunks: int
    processed_files: int


class HealthResponse(BaseModel):
    status: str

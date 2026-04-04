from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    answer_id: Optional[str] = None
    question: str
    answer: str
    citations: List[Citation]
    confidence_note: str
    realtime_flight: Optional["RealtimeFlightCard"] = None
    realtime_flight_details: Optional[Dict[str, Any]] = None


class RealtimeFlightCard(BaseModel):
    flight_no: str
    status: Optional[str] = None
    planned_departure: Optional[str] = None
    actual_departure: Optional[str] = None
    planned_arrival: Optional[str] = None
    actual_arrival: Optional[str] = None
    delay_minutes: Optional[int] = None
    terminal: Optional[str] = None
    gate: Optional[str] = None


class FlightRealtimeRequest(BaseModel):
    question: Optional[str] = None
    flight_no: Optional[str] = None


class AnswerFeedbackRequest(BaseModel):
    answer_id: Optional[str] = None
    question: str = Field(min_length=2)
    answer: str = Field(min_length=1)
    confidence_note: str = Field(min_length=1)
    rating: int = Field(ge=-1, le=1, description="1=like, -1=dislike")
    corrected_answer: Optional[str] = None
    comment: Optional[str] = None


class AnswerFeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    patch_applied: bool = False
    patch_status: Optional[str] = None
    patch_path: Optional[str] = None
    iterated_answer: Optional[str] = None


class IngestRequest(BaseModel):
    input_path: str = Field(description="文档目录或文件路径")


class IngestResponse(BaseModel):
    indexed_chunks: int
    processed_files: int


class HealthResponse(BaseModel):
    status: str

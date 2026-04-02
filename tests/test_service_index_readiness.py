from pathlib import Path
from types import SimpleNamespace

from airport_rag.service import AirportRAGService
from airport_rag.vector_store import RetrievedChunk


class _FakeStore:
    def __init__(self, count_value: int = 0) -> None:
        self._count_value = count_value

    def count(self) -> int:
        return self._count_value

    def query(self, query_embedding, top_k: int, where=None):
        return []


class _FakeEmbedding:
    def embed_query(self, question: str):
        return [0.0, 0.0, 0.0]


def _new_service_for_index_empty() -> AirportRAGService:
    svc = AirportRAGService.__new__(AirportRAGService)
    svc.settings = SimpleNamespace(top_k=5, openai_api_key=None)
    svc.embedding = _FakeEmbedding()
    svc.store = _FakeStore(count_value=0)
    svc._index_rebuild_attempted = True
    return svc


def test_ask_returns_index_empty_when_store_has_no_chunks() -> None:
    svc = _new_service_for_index_empty()

    result = svc.ask("国内出发需要提前多久到达？")

    assert result.confidence_note == "index-empty"
    assert result.citations == []
    assert "知识库索引为空" in result.answer


def test_ensure_index_ready_marks_attempt_when_docs_missing(monkeypatch) -> None:
    svc = _new_service_for_index_empty()
    svc._index_rebuild_attempted = False

    original_exists = Path.exists

    def _fake_exists(self: Path) -> bool:
        if str(self).endswith("data/documents"):
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", _fake_exists)

    svc._ensure_index_ready()

    assert svc._index_rebuild_attempted is True
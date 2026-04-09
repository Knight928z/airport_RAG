from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from airport_rag import api as api_module


class _FakeStore:
    def __init__(self) -> None:
        self._count = 6
        self.reset_called = False

    def count(self) -> int:
        return self._count

    def get_chunks(self, limit: int = 200, offset: int = 0) -> dict:
        ids = ["a1", "a2", "a3", "a4", "a5", "a6"]
        docs = [
            "国际到达流程请按指引前往到达层。",
            "国际到达流程请按指引前往到达层。",
            "短",
            "海关申报请走红色通道。",
            "海关申报请走红色通道。",
            "登机前请核验身份证件。",
        ]
        metas = [
            {"source": "/data/documents/airport/到达指南", "doc_scope": "airport"},
            {"source": "/data/documents/airport/到达指南", "doc_scope": "airport"},
            {"source": "", "doc_scope": "unknown"},
            {"source": "/data/documents/airport/海关须知", "doc_scope": "airport"},
            {"source": "/data/documents/airport/海关须知", "doc_scope": "airport"},
            {"source": "/data/documents/airport/登机须知", "doc_scope": "airport"},
        ]

        sub = list(zip(ids, docs, metas))[offset : offset + limit]
        return {
            "ids": [x[0] for x in sub],
            "documents": [x[1] for x in sub],
            "metadatas": [x[2] for x in sub],
        }

    def reset_collection(self) -> None:
        self.reset_called = True
        self._count = 0


def test_admin_vector_inspect_returns_metrics(monkeypatch) -> None:
    fake_store = _FakeStore()
    monkeypatch.setattr(api_module.service, "store", fake_store)

    client = TestClient(api_module.app)
    resp = client.post("/admin/vector/inspect", json={"sample_limit": 100, "top_duplicates": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    ins = body["inspection"]
    assert ins["total_vectors"] == 6
    assert ins["sample_size"] == 6
    assert ins["duplicate_group_count"] >= 1
    assert ins["duplicate_chunk_count"] >= 2
    assert ins["short_text_count"] >= 1
    assert "recommendations" in ins


def test_admin_vector_rebuild_resets_and_reingests(monkeypatch) -> None:
    fake_store = _FakeStore()
    monkeypatch.setattr(api_module.service, "store", fake_store)

    monkeypatch.setattr(
        api_module.service,
        "ingest",
        lambda input_path: SimpleNamespace(indexed_chunks=12, processed_files=3),
    )

    client = TestClient(api_module.app)
    resp = client.post(
        "/admin/vector/rebuild",
        json={"input_path": "./data/documents", "reset_collection": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["reset_collection"] is True
    assert body["indexed_chunks"] == 12
    assert body["processed_files"] == 3
    assert fake_store.reset_called is True

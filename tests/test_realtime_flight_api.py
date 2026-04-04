from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from airport_rag import api as api_module
from airport_rag.schemas import AskResponse, RealtimeFlightCard


def test_ask_realtime_question_uses_mcp_result(tmp_path: Path, monkeypatch) -> None:
    def _fake_realtime(question: str, flight_no=None):
        return (
            "MU2456 实时状态：预计延误 25 分钟。",
            RealtimeFlightCard(
                flight_no="MU2456",
                status="延误",
                planned_departure="2026-04-04 10:00",
                actual_departure="2026-04-04 10:25",
                planned_arrival="2026-04-04 12:10",
                actual_arrival="2026-04-04 12:35",
                delay_minutes=25,
                terminal="出发 T1 / 到达 T2",
                gate="出发 A12 / 到达 B03",
            ),
        )

    monkeypatch.setattr(api_module, "query_realtime_flight", _fake_realtime)
    doc_root = tmp_path / "documents"
    doc_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_module, "DOC_ROOT", doc_root)

    ingest_calls = {"count": 0}

    class _IngestResult:
        indexed_chunks = 2
        processed_files = 1

    def _fake_ingest(_path: str):
        ingest_calls["count"] += 1
        return _IngestResult()

    monkeypatch.setattr(api_module.service, "ingest", _fake_ingest)
    monkeypatch.setattr(
        api_module.service,
        "ask",
        lambda question, top_k=None: AskResponse(
            question=question,
            answer="fallback",
            citations=[],
            confidence_note="retrieval-extractive",
        ),
    )

    client = TestClient(api_module.app)
    resp = client.post("/ask", json={"question": "MU2456现在延误了吗？"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence_note"] == "realtime-flight"
    assert body["realtime_flight"]["flight_no"] == "MU2456"
    assert body["realtime_flight"]["delay_minutes"] == 25
    assert ingest_calls["count"] == 1

    records = list((doc_root / "airport" / "实时航班").glob("*.md"))
    assert records
    assert "MU2456" in records[0].read_text(encoding="utf-8")


def test_ask_fallback_to_rag_when_realtime_not_matched(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "query_realtime_flight", lambda question, flight_no=None: None)

    monkeypatch.setattr(
        api_module.service,
        "ask",
        lambda question, top_k=None: AskResponse(
            question=question,
            answer="RAG 正常回答",
            citations=[],
            confidence_note="rule-based",
        ),
    )

    client = TestClient(api_module.app)
    resp = client.post("/ask", json={"question": "国内出发需要提前多久到达？"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence_note"] == "rule-based"
    assert body["realtime_flight"] is None


def test_flight_realtime_endpoint_returns_standard_card(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "query_realtime_flight",
        lambda question, flight_no=None: (
            "CA1307 状态正常",
            RealtimeFlightCard(
                flight_no="CA1307",
                status="正常",
                planned_departure="10:30",
                actual_departure="10:32",
                planned_arrival="13:00",
                actual_arrival="13:02",
                delay_minutes=2,
                terminal="T2",
                gate="C18",
            ),
        ),
    )

    client = TestClient(api_module.app)
    resp = client.post("/flight/realtime", json={"flight_no": "CA1307"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence_note"] == "realtime-flight"
    assert body["realtime_flight"]["flight_no"] == "CA1307"
    assert body["realtime_flight"]["gate"] == "C18"

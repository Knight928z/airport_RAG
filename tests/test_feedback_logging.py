from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from airport_rag import api as api_module
from airport_rag.schemas import AskResponse


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_ask_low_confidence_auto_records_uncovered(tmp_path: Path, monkeypatch) -> None:
    feedback_root = tmp_path / "feedback"
    monkeypatch.setattr(api_module, "FEEDBACK_ROOT", feedback_root)
    monkeypatch.setattr(api_module, "ANSWER_FEEDBACK_LOG", feedback_root / "answer_feedback.jsonl")
    monkeypatch.setattr(api_module, "UNCOVERED_LOG", feedback_root / "uncovered_questions.jsonl")

    def _fake_ask(question: str, top_k=None):
        return AskResponse(
            question=question,
            answer="证据不足",
            citations=[],
            confidence_note="low-confidence",
        )

    monkeypatch.setattr(api_module.service, "ask", _fake_ask)

    client = TestClient(api_module.app)
    resp = client.post("/ask", json={"question": "值机柜台几点关闭？"})

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("answer_id")

    uncovered = _read_jsonl(api_module.UNCOVERED_LOG)
    assert len(uncovered) == 1
    assert uncovered[0]["question"] == "值机柜台几点关闭？"
    assert uncovered[0]["reason"] == "auto-low-confidence"


def test_feedback_dislike_and_correction_are_logged(tmp_path: Path, monkeypatch) -> None:
    feedback_root = tmp_path / "feedback"
    monkeypatch.setattr(api_module, "FEEDBACK_ROOT", feedback_root)
    monkeypatch.setattr(api_module, "ANSWER_FEEDBACK_LOG", feedback_root / "answer_feedback.jsonl")
    monkeypatch.setattr(api_module, "UNCOVERED_LOG", feedback_root / "uncovered_questions.jsonl")

    client = TestClient(api_module.app)
    resp = client.post(
        "/feedback",
        json={
            "answer_id": "ans-1",
            "question": "我的充电宝150Wh能带吗？",
            "answer": "当前证据仅明确<=100Wh。",
            "confidence_note": "rule-based",
            "rating": -1,
            "corrected_answer": "按航司规则，100-160Wh通常需航司同意。",
            "comment": "依据应更聚焦锂电池条款",
        },
    )

    assert resp.status_code == 200

    feedback = _read_jsonl(api_module.ANSWER_FEEDBACK_LOG)
    assert len(feedback) == 1
    assert feedback[0]["answer_id"] == "ans-1"
    assert feedback[0]["rating"] == -1
    assert "100-160Wh" in feedback[0]["corrected_answer"]

    uncovered = _read_jsonl(api_module.UNCOVERED_LOG)
    assert len(uncovered) == 1
    assert uncovered[0]["reason"] == "user-dislike"
    assert uncovered[0]["question"] == "我的充电宝150Wh能带吗？"

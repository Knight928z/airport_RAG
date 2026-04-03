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
    monkeypatch.setattr(api_module, "PATCH_REGISTRY_LOG", feedback_root / "patch_registry.jsonl")

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
    doc_root = tmp_path / "documents"
    (doc_root / "airport").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_module, "FEEDBACK_ROOT", feedback_root)
    monkeypatch.setattr(api_module, "ANSWER_FEEDBACK_LOG", feedback_root / "answer_feedback.jsonl")
    monkeypatch.setattr(api_module, "UNCOVERED_LOG", feedback_root / "uncovered_questions.jsonl")
    monkeypatch.setattr(api_module, "PATCH_REGISTRY_LOG", feedback_root / "patch_registry.jsonl")
    monkeypatch.setattr(api_module, "DOC_ROOT", doc_root)

    ingest_calls = {"count": 0}

    class _IngestResult:
        indexed_chunks = 1
        processed_files = 1

    def _fake_ingest(_path: str):
        ingest_calls["count"] += 1
        return _IngestResult()

    def _fake_ask(question: str, top_k=None):
        return AskResponse(
            question=question,
            answer="已根据补丁迭代后的答案",
            citations=[],
            confidence_note="rule-based",
        )

    monkeypatch.setattr(api_module.service, "ingest", _fake_ingest)
    monkeypatch.setattr(api_module.service, "ask", _fake_ask)

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
    body = resp.json()
    assert body["patch_applied"] is True
    assert body["patch_status"] in {"applied", "merged"}
    assert body["patch_path"].endswith("用户纠错补丁.md")
    assert "迭代" in body.get("iterated_answer", "")
    assert ingest_calls["count"] == 1

    feedback = _read_jsonl(api_module.ANSWER_FEEDBACK_LOG)
    assert len(feedback) == 1
    assert feedback[0]["answer_id"] == "ans-1"
    assert feedback[0]["rating"] == -1
    assert "100-160Wh" in feedback[0]["corrected_answer"]

    uncovered = _read_jsonl(api_module.UNCOVERED_LOG)
    assert len(uncovered) == 1
    assert uncovered[0]["reason"] == "user-dislike"
    assert uncovered[0]["question"] == "我的充电宝150Wh能带吗？"

    patch_file = doc_root / body["patch_path"]
    assert patch_file.exists()
    patch_text = patch_file.read_text(encoding="utf-8")
    assert "用户纠错知识补丁" in patch_text
    assert "建议修正" in patch_text


def test_feedback_like_does_not_trigger_patch_flow(tmp_path: Path, monkeypatch) -> None:
    feedback_root = tmp_path / "feedback"
    doc_root = tmp_path / "documents"
    (doc_root / "airport").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_module, "FEEDBACK_ROOT", feedback_root)
    monkeypatch.setattr(api_module, "ANSWER_FEEDBACK_LOG", feedback_root / "answer_feedback.jsonl")
    monkeypatch.setattr(api_module, "UNCOVERED_LOG", feedback_root / "uncovered_questions.jsonl")
    monkeypatch.setattr(api_module, "PATCH_REGISTRY_LOG", feedback_root / "patch_registry.jsonl")
    monkeypatch.setattr(api_module, "DOC_ROOT", doc_root)

    client = TestClient(api_module.app)
    resp = client.post(
        "/feedback",
        json={
            "answer_id": "ans-like-1",
            "question": "国内出发需要提前多久到达？",
            "answer": "建议提前2小时到达。",
            "confidence_note": "grounded-factoid",
            "rating": 1,
            "corrected_answer": "",
            "comment": "很好",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["patch_applied"] is False
    assert body["patch_status"] == "skipped"
    assert body["patch_path"] is None
    assert body["iterated_answer"] is None


def test_feedback_patch_deduplicates_same_question(tmp_path: Path, monkeypatch) -> None:
    feedback_root = tmp_path / "feedback"
    doc_root = tmp_path / "documents"
    (doc_root / "airport").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_module, "FEEDBACK_ROOT", feedback_root)
    monkeypatch.setattr(api_module, "ANSWER_FEEDBACK_LOG", feedback_root / "answer_feedback.jsonl")
    monkeypatch.setattr(api_module, "UNCOVERED_LOG", feedback_root / "uncovered_questions.jsonl")
    monkeypatch.setattr(api_module, "PATCH_REGISTRY_LOG", feedback_root / "patch_registry.jsonl")
    monkeypatch.setattr(api_module, "DOC_ROOT", doc_root)

    class _IngestResult:
        indexed_chunks = 1
        processed_files = 1

    monkeypatch.setattr(api_module.service, "ingest", lambda _path: _IngestResult())
    monkeypatch.setattr(
        api_module.service,
        "ask",
        lambda question, top_k=None: AskResponse(
            question=question,
            answer="迭代答案",
            citations=[],
            confidence_note="rule-based",
        ),
    )

    client = TestClient(api_module.app)
    payload = {
        "answer_id": "ans-dupe-1",
        "question": "可以托运锂电池吗？",
        "answer": "不能确定。",
        "confidence_note": "low-confidence",
        "rating": -1,
        "corrected_answer": "充电宝、锂电池禁止作为行李托运。",
        "comment": "请按危险品规则回答。",
    }
    first = client.post("/feedback", json=payload)
    second = client.post("/feedback", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["patch_applied"] is True
    assert second.json()["patch_applied"] is False
    assert second.json()["patch_status"] == "deduplicated"


def test_feedback_patch_auto_merges_when_threshold_reached(tmp_path: Path, monkeypatch) -> None:
    feedback_root = tmp_path / "feedback"
    doc_root = tmp_path / "documents"
    (doc_root / "airport").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_module, "FEEDBACK_ROOT", feedback_root)
    monkeypatch.setattr(api_module, "ANSWER_FEEDBACK_LOG", feedback_root / "answer_feedback.jsonl")
    monkeypatch.setattr(api_module, "UNCOVERED_LOG", feedback_root / "uncovered_questions.jsonl")
    monkeypatch.setattr(api_module, "PATCH_REGISTRY_LOG", feedback_root / "patch_registry.jsonl")
    monkeypatch.setattr(api_module, "DOC_ROOT", doc_root)
    monkeypatch.setattr(api_module, "PATCH_MERGE_ENTRY_THRESHOLD", 2)
    monkeypatch.setattr(api_module, "PATCH_MAX_BYTES", 10**9)

    class _IngestResult:
        indexed_chunks = 1
        processed_files = 1

    monkeypatch.setattr(api_module.service, "ingest", lambda _path: _IngestResult())
    monkeypatch.setattr(
        api_module.service,
        "ask",
        lambda question, top_k=None: AskResponse(
            question=question,
            answer="迭代答案",
            citations=[],
            confidence_note="rule-based",
        ),
    )

    client = TestClient(api_module.app)
    for i in range(2):
        resp = client.post(
            "/feedback",
            json={
                "answer_id": f"ans-merge-{i}",
                "question": f"测试问题{i}：国内航班能携带液体吗？",
                "answer": "旧答案",
                "confidence_note": "low-confidence",
                "rating": -1,
                "corrected_answer": f"修正答案{i}",
                "comment": "液体规则",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["patch_status"] == "merged"

    merged_doc = doc_root / "airport" / "知识补丁合并稿.md"
    assert merged_doc.exists()
    merged_text = merged_doc.read_text(encoding="utf-8")
    assert "patch-merge" in merged_text

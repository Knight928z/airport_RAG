from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from airport_rag import api as api_module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_admin_patch_stats_returns_topic_and_dedup_metrics(tmp_path: Path, monkeypatch) -> None:
    doc_root = tmp_path / "documents"
    patch_root = tmp_path / "patches"
    feedback_root = tmp_path / "feedback"

    monkeypatch.setattr(api_module, "DOC_ROOT", doc_root)
    monkeypatch.setattr(api_module, "PATCH_ROOT", patch_root)
    monkeypatch.setattr(api_module, "FEEDBACK_ROOT", feedback_root)
    monkeypatch.setattr(api_module, "PATCH_AUDIT_LOG", feedback_root / "patch_audit.jsonl")

    patch_file = patch_root / "airport" / "行李" / "high" / "2026-04-用户纠错补丁.md"
    _write(
        patch_file,
        "# 用户纠错知识补丁\n\n"
        "## feedback-patch 2026-04-03T00:00:00\n- 问题：Q1\n- 建议修正：A1\n"
        "\n\n---\n## feedback-patch 2026-04-03T00:00:01\n- 问题：Q2\n- 建议修正：A2\n",
    )

    audit_rows = [
        {"status": "applied"},
        {"status": "deduplicated"},
        {"status": "merged"},
    ]
    (feedback_root / "patch_audit.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (feedback_root / "patch_audit.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in audit_rows) + "\n",
        encoding="utf-8",
    )

    client = TestClient(api_module.app)
    resp = client.get("/admin/patches/stats")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_active_patch_files"] == 1
    assert data["total_active_patch_entries"] == 2
    assert data["deduplicated_count"] == 1
    assert data["merged_count"] == 1
    assert data["dedup_rate"] > 0
    assert "行李" in data["topic_stats"]


def test_admin_patch_review_merge_cleans_patch_and_writes_main_doc(tmp_path: Path, monkeypatch) -> None:
    doc_root = tmp_path / "documents"
    patch_root = tmp_path / "patches"
    feedback_root = tmp_path / "feedback"

    monkeypatch.setattr(api_module, "DOC_ROOT", doc_root)
    monkeypatch.setattr(api_module, "PATCH_ROOT", patch_root)
    monkeypatch.setattr(api_module, "FEEDBACK_ROOT", feedback_root)
    monkeypatch.setattr(api_module, "PATCH_AUDIT_LOG", feedback_root / "patch_audit.jsonl")

    patch_file = patch_root / "airport" / "行李" / "high" / "2026-04-用户纠错补丁.md"
    _write(
        patch_file,
        "# 用户纠错知识补丁\n\n"
        "## feedback-patch 2026-04-03T00:00:00\n"
        "- 问题：可以托运锂电池吗？\n"
        "- 置信分层：high\n"
        "- 建议修正：不能托运。\n",
    )

    class _IngestResult:
        indexed_chunks = 1
        processed_files = 1

    monkeypatch.setattr(api_module.service, "ingest", lambda _path: _IngestResult())

    client = TestClient(api_module.app)
    resp = client.post("/admin/patches/review-merge", params={"cleanup": "true"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["merged_files"] == 1
    assert body["merged_entries"] >= 1

    merged_doc = doc_root / "airport" / "知识补丁合并稿.md"
    assert merged_doc.exists()
    merged_text = merged_doc.read_text(encoding="utf-8")
    assert "patch-merge" in merged_text

    assert patch_file.exists() is False
    archived = list((patch_file.parent / "archive").glob("*.md"))
    assert archived

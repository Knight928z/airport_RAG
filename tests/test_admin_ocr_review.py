from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from airport_rag import api as api_module


class _FakeIngestResult:
    indexed_chunks = 3
    processed_files = 1


def test_admin_ocr_review_items_lists_sidecars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "DOC_ROOT", tmp_path)

    source = tmp_path / "airport" / "安检须知.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"\x89PNG\r\n\x1a\n")

    sidecar = tmp_path / "airport" / "安检须知.png.ocr.md"
    sidecar.write_text("# OCR 文本\n\n禁止携带超限液体。\n", encoding="utf-8")

    client = TestClient(api_module.app)
    resp = client.get("/admin/ocr-review/items")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1

    item = body["items"][0]
    assert item["ocr_path"] == "airport/安检须知.png.ocr.md"
    assert item["source_path"] == "airport/安检须知.png"
    assert item["source_exists"] is True
    assert item["chars"] > 0
    assert item["source_preview_url"].startswith("/admin/docs/raw?path=")


def test_admin_ocr_review_update_content_can_sync(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "DOC_ROOT", tmp_path)
    monkeypatch.setattr(api_module.service, "ingest", lambda *_args, **_kwargs: _FakeIngestResult())

    sidecar = tmp_path / "airport" / "值机提示.jpg.ocr.md"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("旧文本", encoding="utf-8")

    client = TestClient(api_module.app)
    resp = client.put(
        "/admin/ocr-review/content",
        params={"path": "airport/值机提示.jpg.ocr.md"},
        json={"content": "新文本：请提前90分钟到达。", "auto_sync": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "airport/值机提示.jpg.ocr.md"
    assert body["synced"] is True
    assert body["indexed_chunks"] == 3

    assert sidecar.read_text(encoding="utf-8") == "新文本：请提前90分钟到达。"


def test_admin_ocr_review_update_rejects_non_sidecar_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "DOC_ROOT", tmp_path)

    target = tmp_path / "airport" / "普通文档.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello", encoding="utf-8")

    client = TestClient(api_module.app)
    resp = client.put(
        "/admin/ocr-review/content",
        params={"path": "airport/普通文档.md"},
        json={"content": "changed", "auto_sync": False},
    )

    assert resp.status_code == 400
    assert "ocr.md" in resp.json()["detail"]

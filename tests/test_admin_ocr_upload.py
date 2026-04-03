from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from airport_rag import api as api_module


class _FakeIngestResult:
    indexed_chunks = 0
    processed_files = 0


def test_admin_bulk_upload_image_creates_ocr_sidecar(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "DOC_ROOT", tmp_path)
    monkeypatch.setattr(api_module.service, "ingest", lambda *_args, **_kwargs: _FakeIngestResult())
    monkeypatch.setattr(api_module, "extract_text_from_image", lambda _p: "这是图片OCR结果")

    client = TestClient(api_module.app)

    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    resp = client.post(
        "/admin/docs/bulk",
        data={"auto_sync": "false"},
        files=[("files", ("安检告示.png", png_bytes, "image/png"))],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["created"]) == 1

    created = body["created"][0]
    assert created["path"].endswith("安检告示.png")
    assert created["ocr_text_path"] is not None
    assert created["ocr_text_path"].endswith("安检告示.png.ocr.md")
    assert created["ocr_chars"] > 0

    ocr_file = tmp_path / created["ocr_text_path"]
    assert ocr_file.exists()
    assert "OCR结果" in ocr_file.read_text(encoding="utf-8")

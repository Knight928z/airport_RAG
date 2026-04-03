from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from airport_rag import api as api_module


class _FakeIngestResult:
    indexed_chunks = 0
    processed_files = 0


def test_admin_bulk_upload_keeps_pdf_binary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "DOC_ROOT", tmp_path)
    monkeypatch.setattr(api_module.service, "ingest", lambda *_args, **_kwargs: _FakeIngestResult())

    client = TestClient(api_module.app)

    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"

    resp = client.post(
        "/admin/docs/bulk",
        data={"auto_sync": "false"},
        files=[("files", ("测试航司文档.pdf", pdf_bytes, "application/pdf"))],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["created"]) == 1
    rel_path = data["created"][0]["path"]

    saved = (tmp_path / rel_path).read_bytes()
    assert saved == pdf_bytes


def test_admin_get_doc_content_marks_pdf_binary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "DOC_ROOT", tmp_path)

    p = tmp_path / "airport" / "说明.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.4\n%%EOF\n")

    client = TestClient(api_module.app)
    resp = client.get("/admin/docs/content", params={"path": "airport/说明.pdf"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["is_binary"] is True
    assert data["editable"] is False
    assert data["preview_url"].startswith("/admin/docs/raw?path=")


def test_admin_get_doc_raw_returns_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "DOC_ROOT", tmp_path)

    p = tmp_path / "airport" / "说明.pdf"
    p.parent.mkdir(parents=True, exist_ok=True)
    content = b"%PDF-1.4\n%%EOF\n"
    p.write_bytes(content)

    client = TestClient(api_module.app)
    resp = client.get("/admin/docs/raw", params={"path": "airport/说明.pdf"})

    assert resp.status_code == 200
    assert resp.content == content
    disposition = resp.headers.get("content-disposition", "").lower()
    assert "inline" in disposition
    assert "attachment" not in disposition

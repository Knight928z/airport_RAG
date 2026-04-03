from __future__ import annotations

from airport_rag.ingest import ingest_path, load_documents


class _FakeStore:
    def __init__(self):
        self.last_metadatas = []

    def add_chunks(self, ids, texts, embeddings, metadatas):
        assert len(ids) == len(texts) == len(embeddings) == len(metadatas)
        assert all(meta.get("page") is not None for meta in metadatas)
        self.last_metadatas = metadatas
        return len(ids)


class _FakeEmbedding:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_load_documents_supports_extensionless_text_file(tmp_path) -> None:
    doc = tmp_path / "托运行李规定"
    doc.write_text("托运行李须符合尺寸与重量限制。", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].source.endswith("托运行李规定")
    assert "托运行李" in docs[0].text


def test_ingest_path_with_extensionless_file(tmp_path) -> None:
    doc = tmp_path / "出发指南-国内出发"
    doc.write_text("请至少提前90分钟到达航站楼办理值机。", encoding="utf-8")

    indexed_chunks, processed_files = ingest_path(
        str(tmp_path),
        store=_FakeStore(),
        embedding=_FakeEmbedding(),
    )

    assert processed_files == 1
    assert indexed_chunks >= 1


def test_ingest_path_sets_scope_metadata_from_documents_hierarchy(tmp_path) -> None:
    airport_doc = tmp_path / "data" / "documents" / "airport" / "出发指南-国内出发"
    airport_doc.parent.mkdir(parents=True, exist_ok=True)
    airport_doc.write_text("国内出发建议提前2小时到达。", encoding="utf-8")

    cz_doc = tmp_path / "data" / "documents" / "CZ" / "南航行李规定.md"
    cz_doc.parent.mkdir(parents=True, exist_ok=True)
    cz_doc.write_text("南航托运行李按航司规定执行。", encoding="utf-8")

    store = _FakeStore()
    indexed_chunks, processed_files = ingest_path(
        str(tmp_path / "data" / "documents"),
        store=store,
        embedding=_FakeEmbedding(),
    )

    assert processed_files == 2
    assert indexed_chunks >= 2

    scopes = {(meta.get("doc_scope"), meta.get("carrier")) for meta in store.last_metadatas}
    assert ("airport", "") in scopes
    assert ("airline", "CZ") in scopes


def test_load_documents_supports_image_ocr(monkeypatch, tmp_path) -> None:
    img = tmp_path / "安检提醒.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr("airport_rag.ingest.extract_text_from_image", lambda _p: "图片OCR文本")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].source.endswith("安检提醒.png")
    assert "OCR" in docs[0].text

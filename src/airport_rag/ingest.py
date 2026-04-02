from __future__ import annotations

import pathlib
import re
import uuid
from dataclasses import dataclass

from .embeddings import EmbeddingProvider
from .langchain_utils import split_text_with_langchain
from .vector_store import ChromaStore


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class RawDoc:
    text: str
    source: str
    page: int | None = None


def split_text(text: str, chunk_size: int = 280, overlap: int = 60) -> list[str]:
    return split_text_with_langchain(text, chunk_size=chunk_size, overlap=overlap)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _sliding_split(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_documents(path: str) -> list[RawDoc]:
    root = pathlib.Path(path)
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and _is_supported_file(p)]
    docs: list[RawDoc] = []

    for file_path in files:
        ext = file_path.suffix.lower()
        if ext in {".txt", ".md"} or ext == "":
            text = _read_text_file(file_path)
            if text:
                docs.append(RawDoc(text=text, source=str(file_path)))
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(file_path))
                for page_number, page in enumerate(reader.pages, start=1):
                    docs.append(
                        RawDoc(
                            text=page.extract_text() or "",
                            source=str(file_path),
                            page=page_number,
                        )
                    )
            except Exception:
                continue
    return docs


def _is_supported_file(file_path: pathlib.Path) -> bool:
    if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return True
    if file_path.suffix == "" and not file_path.name.startswith("."):
        return True
    return False


def _read_text_file(file_path: pathlib.Path) -> str:
    raw = file_path.read_bytes()
    if b"\x00" in raw:
        return ""
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _derive_source_profile(source: str) -> tuple[str, str]:
    normalized = source.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    lowered = [p.lower() for p in parts]

    if "documents" in lowered:
        idx = lowered.index("documents")
        if idx + 1 < len(parts):
            folder = parts[idx + 1]
            if folder.lower() == "airport":
                return "airport", ""
            return "airline", folder.upper()

    return "unknown", ""


def ingest_path(path: str, store: ChromaStore, embedding: EmbeddingProvider) -> tuple[int, int]:
    docs = load_documents(path)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for raw in docs:
        doc_scope, carrier = _derive_source_profile(raw.source)
        pieces = split_text(raw.text)
        for i, piece in enumerate(pieces):
            cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{raw.source}|{raw.page}|{i}|{piece[:60]}"))
            ids.append(cid)
            texts.append(piece)
            metadatas.append(
                {
                    "source": raw.source,
                    "page": raw.page if raw.page is not None else -1,
                    "doc_scope": doc_scope,
                    "carrier": carrier,
                }
            )

    embeddings = embedding.embed_documents(texts)
    count = store.add_chunks(ids, texts, embeddings, metadatas)
    return count, len({d.source for d in docs})

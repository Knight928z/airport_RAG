from __future__ import annotations

from typing import List


def split_text_with_langchain(text: str, chunk_size: int = 280, overlap: int = 60) -> List[str]:
    """Split text with LangChain's RecursiveCharacterTextSplitter.

    Falls back to a simple sliding-window split when LangChain is unavailable.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.strip() for line in normalized.split("\n") if line.strip())
    if not normalized:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
        )
        chunks = [c.strip() for c in splitter.split_text(normalized) if c.strip()]
        if chunks:
            return chunks
    except Exception:
        pass

    # fallback
    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks

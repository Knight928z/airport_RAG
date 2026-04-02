from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page: int | None
    distance: float
    doc_scope: str = "unknown"
    carrier: str = ""


class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        try:
            import chromadb
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("缺少 chromadb 依赖，请先安装 requirements.txt") from exc

        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def add_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> int:
        if not ids:
            return 0
        try:
            self._collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        except Exception as exc:
            if self._is_dimension_mismatch(exc):
                self._recreate_collection()
                self._collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
            else:
                raise
        return len(ids)

    def query(self, query_embedding: list[float], top_k: int, where: dict | None = None) -> list[RetrievedChunk]:
        total = self._collection.count()
        if total <= 0:
            return []

        n_results = min(top_k, total)
        try:
            kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
            if where:
                kwargs["where"] = where
            result = self._collection.query(**kwargs)
        except Exception as exc:
            if self._is_dimension_mismatch(exc):
                self._recreate_collection()
                return []
            raise
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            page_value = meta.get("page")
            page = None if page_value in (-1, "-1", None) else int(page_value)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=meta.get("source", "unknown"),
                    page=page,
                    distance=float(dist),
                    doc_scope=str(meta.get("doc_scope", "unknown") or "unknown"),
                    carrier=str(meta.get("carrier", "") or ""),
                )
            )
        return chunks

    def count(self) -> int:
        return int(self._collection.count())

    def _is_dimension_mismatch(self, exc: Exception) -> bool:
        return "does not match collection dimensionality" in str(exc).lower()

    def _recreate_collection(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

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
            if self._is_hnsw_query_instability(exc):
                result = self._retry_query_with_fallback(query_embedding, n_results, where)
                if result is None:
                    return []
            else:
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

    def get_chunks(self, limit: int = 200, offset: int = 0) -> dict:
        safe_limit = max(1, int(limit))
        safe_offset = max(0, int(offset))
        return self._collection.get(
            limit=safe_limit,
            offset=safe_offset,
            include=["documents", "metadatas"],
        )

    def reset_collection(self) -> None:
        self._recreate_collection()

    def _is_dimension_mismatch(self, exc: Exception) -> bool:
        return "does not match collection dimensionality" in str(exc).lower()

    def _is_hnsw_query_instability(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "contigious 2d array" in text or "contiguous 2d array" in text or "ef or m is too small" in text

    def _retry_query_with_fallback(self, query_embedding: list[float], n_results: int, where: dict | None) -> dict | None:
        attempts = []
        for candidate_k in [n_results, min(n_results, 20), min(n_results, 10), min(n_results, 5), 1]:
            if candidate_k > 0:
                attempts.append((candidate_k, where))
        if where is not None:
            for candidate_k in [min(n_results, 20), min(n_results, 10), min(n_results, 5), 1]:
                if candidate_k > 0:
                    attempts.append((candidate_k, None))

        seen = set()
        for candidate_k, candidate_where in attempts:
            key = (candidate_k, str(candidate_where))
            if key in seen:
                continue
            seen.add(key)
            try:
                kwargs = {"query_embeddings": [query_embedding], "n_results": candidate_k}
                if candidate_where:
                    kwargs["where"] = candidate_where
                return self._collection.query(**kwargs)
            except Exception:
                continue
        return None

    def _recreate_collection(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

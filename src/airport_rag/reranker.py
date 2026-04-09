from __future__ import annotations

import os
import re
from hashlib import md5

from .vector_store import RetrievedChunk


class RerankerProvider:
    """Optional model-based reranker with safe heuristic fallback."""

    def __init__(self, backend: str, model_name: str) -> None:
        self.backend = backend
        self.model_name = model_name
        self._model = None

    def _load_cross_encoder(self) -> bool:
        if self._model is not None:
            return True

        allow_download = os.getenv("RAG_ALLOW_MODEL_DOWNLOAD", "false").lower() in {"1", "true", "yes", "on"}
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, local_files_only=not allow_download)
            return True
        except Exception:
            self._model = None
            return False

    def rerank(self, question: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if len(retrieved) <= 1:
            return retrieved

        if self.backend == "cross_encoder" and self._load_cross_encoder():
            try:
                scores = self.score_pairs(question, [item.text for item in retrieved])
                ranked = sorted(
                    zip(retrieved, scores),
                    key=lambda p: (float(p[1]), -p[0].distance),
                    reverse=True,
                )
                return [item for item, _ in ranked]
            except Exception:
                pass

        return heuristic_rerank(question, retrieved)

    def score_pairs(self, question: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []

        if self.backend == "cross_encoder" and self._load_cross_encoder():
            try:
                pairs = [[question, c] for c in candidates]
                scores = self._model.predict(pairs)
                return [float(s) for s in scores]
            except Exception:
                pass

        # fallback heuristic score aligned with heuristic_rerank behavior
        q_tokens = _tokenize(question)
        vals: list[float] = []
        for c in candidates:
            overlap = len(q_tokens.intersection(_tokenize(c)))
            key_overlap = _keyword_overlap(question, c)
            vals.append(float(overlap + key_overlap))
        return vals


def heuristic_rerank(question: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if len(retrieved) <= 1:
        return retrieved

    q_tokens = _tokenize(question)

    def _score(item: RetrievedChunk) -> tuple[int, int, float]:
        overlap = len(q_tokens.intersection(_tokenize(item.text)))
        key_overlap = _keyword_overlap(question, item.text)
        return overlap, key_overlap, -item.distance

    return sorted(retrieved, key=_score, reverse=True)


def _tokenize(text: str) -> set[str]:
    lowered = (text or "").lower()
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    words = re.findall(r"[A-Za-z0-9_]+", lowered)
    return set(cjk_chars + words)


def _keyword_overlap(question: str, text: str) -> int:
    q = (question or "").lower()
    t = (text or "").lower()
    score = 0
    for kw in ["海关", "边检", "托运", "行李", "充电宝", "锂电池", "值机", "航班", "到达", "出发"]:
        if kw in q and kw in t:
            score += 2
    return score


def reranker_signature(backend: str, model_name: str) -> str:
    raw = f"{backend}|{model_name}"
    return md5(raw.encode("utf-8")).hexdigest()[:10]

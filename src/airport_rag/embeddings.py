from __future__ import annotations

import math
import os
import re
from hashlib import md5


class EmbeddingProvider:
    def __init__(self, backend: str, model_name: str, dim: int = 384) -> None:
        self.backend = backend
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _load_sentence_model(self) -> bool:
        if self._model is not None:
            return True
        allow_download = os.getenv("RAG_ALLOW_MODEL_DOWNLOAD", "false").lower() in {"1", "true", "yes", "on"}
        try:
            from sentence_transformers import SentenceTransformer

            # Prefer local cached model for fast startup/stable offline behavior.
            self._model = SentenceTransformer(self.model_name, local_files_only=not allow_download)
            return True
        except Exception:
            self.backend = "hashing"
            self._model = None
            return False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "sentence_transformers" and self._load_sentence_model():
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [list(map(float, row)) for row in vectors]
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.backend == "sentence_transformers" and self._load_sentence_model():
            vector = self._model.encode([text], normalize_embeddings=True)[0]
            return list(map(float, vector))
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        vec = [0.0] * self.dim
        for token in tokens:
            idx = int(md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

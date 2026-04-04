from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

_DEFAULT_PRIVATE_ENV = Path(__file__).resolve().parents[2] / "data" / "private" / "local.secrets.env"
_private_env_path = os.getenv("RAG_PRIVATE_SECRETS_FILE", str(_DEFAULT_PRIVATE_ENV))
if _private_env_path:
    load_dotenv(dotenv_path=_private_env_path, override=False)


@dataclass
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_backend: str = os.getenv("RAG_EMBEDDING_BACKEND", "sentence_transformers")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    vector_dir: str = os.getenv("RAG_VECTOR_DIR", "./data/chroma")
    collection_name: str = os.getenv("RAG_COLLECTION", "airport_kb")
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))


def get_settings() -> Settings:
    return Settings()

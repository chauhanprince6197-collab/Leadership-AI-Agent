"""
config/settings.py — Centralised configuration (OpenAI version)
"""

from __future__ import annotations
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── OpenAI ────────────────────────────────────────────────────
    openai_api_key: str  = Field(..., description="OpenAI API key")
    openai_model: str    = Field("gpt-4o")          # or "gpt-4o-mini", "gpt-3.5-turbo"
    llm_max_tokens: int  = Field(1024, ge=256, le=4096)
    llm_temperature: float = Field(0.0, ge=0.0, le=1.0)

    # ── Embeddings ─────────────────────────────────────────────────
    embedding_model: str = Field("all-MiniLM-L6-v2")

    # ── Vector store ───────────────────────────────────────────────
    chroma_persist_dir: Path = Field(Path("./chroma_db"))
    chroma_collection: str   = Field("leadership_docs")

    # ── Chunking ───────────────────────────────────────────────────
    chunk_size: int    = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(200, ge=0)

    # ── Retrieval ──────────────────────────────────────────────────
    retrieval_top_k: int  = Field(6, ge=1, le=20)
    retrieval_mode: str   = Field("hybrid")
    mmr_lambda: float     = Field(0.5, ge=0.0, le=1.0)
    rrf_k: int            = Field(60, ge=1)

    # ── Documents ──────────────────────────────────────────────────
    documents_dir: Path = Field(Path("./documents"))
    max_upload_mb: int  = Field(50, ge=1, le=500)

    # ── Server ─────────────────────────────────────────────────────
    host: str  = Field("0.0.0.0")
    port: int  = Field(5050, ge=1024, le=65535)
    debug: bool = Field(False)
    log_level: str = Field("INFO")

    # ── Rate limiting ──────────────────────────────────────────────
    rate_limit_per_minute: int = Field(30, ge=1)

    @field_validator("retrieval_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"dense", "sparse", "hybrid"}
        if v not in allowed:
            raise ValueError(f"retrieval_mode must be one of {allowed}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()

"""
app/__init__.py — Application Factory

FIX SUMMARY (was 5 bytes / empty):
  1. Wires HybridVectorStore into routes module
  2. Applies flask-limiter to all expensive endpoints
  3. Auto-ingests documents folder on first run
  4. Registers both insight and autonomous-agent blueprints
  5. Serves the frontend static HTML
"""

from __future__ import annotations
from pathlib import Path

import structlog
from flask import Flask, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config.settings import settings
from app.retrieval.store import HybridVectorStore
from app.ingestion.pipeline import ingest_folder
import app.api.routes as routes_module
from app.api.routes import api, register_error_handlers

logger = structlog.get_logger(__name__)


def create_app() -> Flask:
    """
    Application factory — call once at startup.

    gunicorn usage:
        gunicorn "main:create_application()" --workers 4 --bind 0.0.0.0:5050
    """
    app = Flask(__name__, static_folder=str(Path(__file__).parent.parent / "frontend"))

    # ── Rate limiting ────────────────────────────────────────────────────
    # storage_uri="memory://" works for single-worker dev.
    # For multi-worker production use Redis:
    #   storage_uri="redis://localhost:6379"
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[f"{settings.rate_limit_per_minute} per minute"],
        storage_uri="memory://",
    )

    # Apply rate limit explicitly to the two most expensive endpoints
    limiter.limit(f"{settings.rate_limit_per_minute} per minute")(routes_module.ask)
    limiter.limit("10 per minute")(routes_module.agent_run)   # agent is heavier

    # ── Initialise vector store (singleton shared across all requests) ───
    store = HybridVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
    )

    # Auto-ingest documents folder on first run (empty store only)
    if store.get_stats()["total_chunks"] == 0:
        logger.info("first_run_ingesting", folder=str(settings.documents_dir))
        chunks = ingest_folder(
            settings.documents_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if chunks:
            store.add_chunks(chunks)
            logger.info("initial_ingestion_complete", total_chunks=len(chunks))
        else:
            logger.warning("no_documents_found", folder=str(settings.documents_dir))
    else:
        logger.info("store_loaded", chunks=store.get_stats()["total_chunks"])

    # Inject store singleton into routes (avoids passing through request context)
    routes_module._store = store

    # ── Register blueprints ──────────────────────────────────────────────
    app.register_blueprint(api)
    register_error_handlers(app)

    # ── Serve frontend ───────────────────────────────────────────────────
    frontend_dir = Path(__file__).parent.parent / "frontend"

    @app.route("/")
    def index():
        return send_from_directory(str(frontend_dir), "index.html")

    logger.info("app_created",
                host=settings.host,
                port=settings.port,
                model=settings.openai_model,
                rate_limit=settings.rate_limit_per_minute)
    return app

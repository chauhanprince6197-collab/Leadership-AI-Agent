"""
app/__init__.py — Application Factory

The factory pattern (create_app) is required for production Flask apps:
  - Enables multiple instances for testing without shared state
  - Makes configuration injectable
  - Required for gunicorn / uWSGI multi-worker deployments
"""

from __future__ import annotations
from pathlib import Path

import structlog
from flask import Flask
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
    Create and configure the Flask application.

    Call once at startup:
        app = create_app()
        app.run(...)
    """
    app = Flask(__name__, static_folder="../frontend")

    # ── Rate limiting ──────────────────────────────────────────────
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[f"{settings.rate_limit_per_minute} per minute"],
        storage_uri="memory://",    # use Redis in prod: "redis://localhost:6379"
    )
    # Apply limiter to expensive endpoints
    limiter.limit(f"{settings.rate_limit_per_minute} per minute")(routes_module.ask)

    # ── Initialise vector store ────────────────────────────────────
    store = HybridVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
    )

    # Ingest documents folder if store is empty (first run)
    if store.get_stats()["total_chunks"] == 0:
        logger.info("first_run_ingesting_documents",
                    folder=str(settings.documents_dir))
        chunks = ingest_folder(
            settings.documents_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if chunks:
            store.add_chunks(chunks)
            logger.info("initial_ingestion_complete",
                        total_chunks=len(chunks))
        else:
            logger.warning("no_documents_found",
                           folder=str(settings.documents_dir))
    else:
        logger.info("store_loaded",
                    chunks=store.get_stats()["total_chunks"])

    # Inject store into routes module (singleton)
    routes_module._store = store

    # ── Register blueprints ────────────────────────────────────────
    app.register_blueprint(api)
    register_error_handlers(app)

    # ── Static frontend ────────────────────────────────────────────
    @app.route("/")
    def index():
        from flask import send_from_directory
        return send_from_directory("../frontend", "index.html")

    logger.info("app_created",
                host=settings.host,
                port=settings.port,
                model=settings.anthropic_model)
    return app

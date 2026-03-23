"""
app/api/routes.py — Production Flask API
 
Includes:
  - Task 1: /api/ask          — RAG-based insight agent
  - Task 2: /api/agent/run    — Autonomous multi-step decision agent
  - Security: API key validation, rate limiting, file type checks
  - Streaming support on both endpoints
"""
 
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional
 
import structlog
from flask import Flask, Blueprint, request, jsonify, Response, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import BaseModel, Field, field_validator
from werkzeug.utils import secure_filename
 
from app.ingestion.pipeline import ingest_file
from app.retrieval.store import HybridVectorStore
from app.retrieval.router import route as auto_route
from app.generation.chain import generate_answer, stream_answer
from app.agent.autonomous import AutonomousDecisionAgent
from config.settings import settings
 
logger = structlog.get_logger(__name__)
 
api = Blueprint("api", __name__, url_prefix="/api")
 
 
# ── Request/Response Models ────────────────────────────────────────────────────
 
class AskRequest(BaseModel):
    question:   str            = Field(..., min_length=3, max_length=2000)
    api_key:    Optional[str]  = Field(None)
    mode:       str            = Field("hybrid")
    where:      Optional[dict] = Field(None)
    auto_route: bool           = Field(True)
    top_k:      int            = Field(6, ge=1, le=20)
    stream:     bool           = Field(False)
 
    @field_validator("mode")
    @classmethod
    def valid_mode(cls, v: str) -> str:
        if v not in {"dense", "sparse", "hybrid"}:
            raise ValueError("mode must be dense, sparse, or hybrid")
        return v
 
    def resolved_api_key(self) -> str:
        key = (self.api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "No API key provided. Enter your Anthropic key in the sidebar "
                "(starts with sk-ant-)."
            )
        # Catch accidental OpenAI key
        if key.startswith("sk-proj-") or (
            key.startswith("sk-") and not key.startswith("sk-ant-")
        ):
            raise ValueError(
                "This looks like an OpenAI key, not an Anthropic key. "
                "This app uses Anthropic Claude. "
                "Get your key at https://console.anthropic.com/settings/keys "
                "(starts with sk-ant-)."
            )
        return key
 
 
class RetrieveRequest(BaseModel):
    query: str            = Field(..., min_length=1, max_length=1000)
    top_k: int            = Field(5, ge=1, le=20)
    where: Optional[dict] = Field(None)
    mode:  str            = Field("hybrid")
 
 
# ── Upload Validation ──────────────────────────────────────────────────────────
 
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md", ".docx", ".html"}
MAX_UPLOAD_BYTES   = settings.max_upload_mb * 1024 * 1024
 
 
def _validate_upload(f) -> tuple[bool, str]:
    filename = secure_filename(f.filename or "")
    if not filename:
        return False, "Invalid filename."
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    if size == 0:
        return False, "File is empty."
    if size > MAX_UPLOAD_BYTES:
        return False, (
            f"File too large ({size/1024/1024:.1f} MB). "
            f"Max: {settings.max_upload_mb} MB."
        )
    return True, ""
 
 
def _friendly_error(err: str) -> str:
    """Convert raw API error messages into user-friendly text."""
    if "429" in err or "quota" in err.lower() or "rate" in err.lower():
        return (
            "Rate limit or quota exceeded on your Anthropic account. "
            "Check https://console.anthropic.com/settings/billing"
        )
    if "401" in err or "authentication" in err.lower():
        return (
            "Invalid Anthropic API key. Make sure it starts with sk-ant- "
            "and was copied correctly from "
            "https://console.anthropic.com/settings/keys"
        )
    if "openai" in err.lower() or "insufficient_quota" in err.lower():
        return (
            "An OpenAI key was used but this app needs an Anthropic key. "
            "Get yours at https://console.anthropic.com/settings/keys "
            "(starts with sk-ant-)"
        )
    return err
 
 
# ── Store Singleton ────────────────────────────────────────────────────────────
 
_store: HybridVectorStore | None = None
 
 
def get_store() -> HybridVectorStore:
    if _store is None:
        raise RuntimeError("Store not initialised. Call create_app() first.")
    return _store
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Leadership Insight Agent (RAG)
# ══════════════════════════════════════════════════════════════════════════════
 
@api.route("/health")
def health():
    """Kubernetes / load-balancer health probe."""
    try:
        stats = get_store().get_stats()
        return jsonify({
            "status":    "healthy",
            "chunks":    stats["total_chunks"],
            "documents": stats["total_documents"],
            "timestamp": int(time.time()),
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503
 
 
@api.route("/status")
def status():
    stats = get_store().get_stats()
    return jsonify({"status": "ready", **stats})
 
 
@api.route("/documents")
def list_documents():
    docs_dir = settings.documents_dir
    files = []
    if docs_dir.exists():
        for fp in sorted(docs_dir.iterdir()):
            if fp.suffix.lower() in ALLOWED_EXTENSIONS:
                files.append({
                    "name":    fp.name,
                    "size_kb": round(fp.stat().st_size / 1024, 1),
                    "ext":     fp.suffix.lower(),
                })
    return jsonify({"documents": files})
 
 
@api.route("/ask", methods=["POST"])
def ask():
    """Task 1: Answer a leadership question using RAG."""
    try:
        req     = AskRequest.model_validate(request.get_json(force=True) or {})
        api_key = req.resolved_api_key()
    except Exception as e:
        return jsonify({"error": str(e)}), 400
 
    params = auto_route(req.question, default_top_k=req.top_k) if req.auto_route else None
    where  = req.where or (params.where if params else None)
    mode   = params.mode  if params else req.mode
    top_k  = params.top_k if params else req.top_k
 
    logger.info("ask_request",
                question_preview=req.question[:80],
                mode=mode, auto_route=req.auto_route, where=where)
 
    try:
        chunks = get_store().query(
            query_text=req.question,
            top_k=top_k,
            where=where,
            mode=mode,
        )
    except Exception as e:
        logger.error("retrieval_failed", error=str(e))
        return jsonify({"error": f"Retrieval error: {e}"}), 500
 
    if not chunks:
        return jsonify({
            "answer": (
                "No relevant content found in the loaded documents. "
                "Make sure your documents have been ingested — use the Upload button "
                "or run: python scripts/ingest.py --folder ./documents"
            ),
            "sources": [], "chunks_retrieved": 0, "retrieval_details": [],
        })
 
    if req.stream:
        def _stream_task1():
            try:
                for token in stream_answer(
                    req.question, chunks, api_key,
                    model=settings.anthropic_model
                ):
                    yield token
            except Exception as e:
                yield f"\n\n[Error: {_friendly_error(str(e))}]"
        return Response(
            stream_with_context(_stream_task1()),
            mimetype="text/plain",
            headers={"X-Sources": ",".join({c["metadata"]["source"] for c in chunks})},
        )
 
    try:
        answer = generate_answer(
            question=req.question,
            chunks=chunks,
            api_key=api_key,
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
        )
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        return jsonify({"error": _friendly_error(str(e))}), 500
 
    return jsonify({
        "answer":           answer,
        "sources":          list({c["metadata"]["source"] for c in chunks}),
        "chunks_retrieved": len(chunks),
        "retrieval_details": [
            {
                "source":       c["metadata"]["source"],
                "doc_type":     c["metadata"]["doc_type"],
                "section":      c["metadata"].get("section", ""),
                "fused_score":  c["retrieval"]["fused_score"],
                "dense_score":  c["retrieval"]["dense_score"],
                "sparse_score": c["retrieval"]["sparse_score"],
                "mode":         c["retrieval"]["mode"],
                "has_numbers":  c["metadata"].get("has_numbers", False),
            }
            for c in chunks
        ],
    })
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Autonomous Decision Agent
# ══════════════════════════════════════════════════════════════════════════════
 
@api.route("/agent/run", methods=["POST"])
def agent_run():
    """
    Task 2: Autonomous multi-step decision agent.
 
    Pipeline:
      1. PLAN     — Claude decomposes the question into 3-5 sub-questions
      2. RESEARCH — RAG retrieval runs independently per sub-question
      3. SYNTHESISE — Claude generates a structured executive decision brief
 
    Supports streaming (stream=true) for live progress updates.
 
    Request body: same as /api/ask
    Response includes: decision_brief, plan, research_steps, sources
    """
    try:
        req     = AskRequest.model_validate(request.get_json(force=True) or {})
        api_key = req.resolved_api_key()
    except Exception as e:
        return jsonify({"error": str(e)}), 400
 
    logger.info("agent_request", question=req.question[:80], stream=req.stream)
 
    try:
        agent = AutonomousDecisionAgent(
            store      = get_store(),
            api_key    = api_key,
            model      = settings.anthropic_model,
            max_tokens = settings.llm_max_tokens,
            top_k      = req.top_k,
        )
 
        # ── Streaming path ────────────────────────────────────────
        if req.stream:
            def _stream_agent():
                try:
                    for token in agent.stream(req.question):
                        yield token
                except Exception as e:
                    yield f"\n\n[Agent error: {_friendly_error(str(e))}]"
            return Response(
                stream_with_context(_stream_agent()),
                mimetype="text/plain",
            )
 
        # ── Standard path ─────────────────────────────────────────
        result = agent.run(req.question)
 
        return jsonify({
            "decision_brief": result.decision_brief,
            "plan":           result.plan,
            "research_steps": [
                {
                    "sub_question": step.sub_question,
                    "answer":       step.answer,
                    "sources":      step.sources,
                    "chunks":       len(step.chunks),
                    "duration_s":   step.duration_s,
                }
                for step in result.research_steps
            ],
            "sources":      result.all_sources,
            "total_chunks": result.total_chunks,
            "duration_s":   result.total_duration,
        })
 
    except Exception as e:
        logger.error("agent_failed", error=str(e))
        return jsonify({"error": _friendly_error(str(e))}), 500
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Shared utility endpoints
# ══════════════════════════════════════════════════════════════════════════════
 
@api.route("/retrieve", methods=["POST"])
def retrieve():
    """Debug: returns raw retrieved chunks without calling the LLM."""
    try:
        req = RetrieveRequest.model_validate(request.get_json(force=True) or {})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
 
    chunks = get_store().query(
        query_text=req.query, top_k=req.top_k,
        where=req.where, mode=req.mode,
    )
    return jsonify({"chunks": [
        {"text": c["text"][:400], "metadata": c["metadata"],
         "retrieval": c["retrieval"]}
        for c in chunks
    ]})
 
 
@api.route("/ingest", methods=["POST"])
def ingest():
    """Upload and ingest a document into the vector store."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
 
    f = request.files["file"]
    ok, err = _validate_upload(f)
    if not ok:
        return jsonify({"error": err}), 400
 
    filename  = secure_filename(f.filename)
    save_path = settings.documents_dir / filename
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    f.save(str(save_path))
 
    try:
        chunks = ingest_file(
            save_path,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        get_store().add_chunks(chunks)
        stats = get_store().get_stats()
        logger.info("file_ingested_via_api", file=filename, new_chunks=len(chunks))
        return jsonify({
            "message": f"'{filename}' ingested — {len(chunks)} chunks added.",
            "stats":   stats,
        })
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error("ingest_failed", file=filename, error=str(e))
        return jsonify({"error": str(e)}), 500
 
 
@api.route("/documents/<source_name>", methods=["DELETE"])
def delete_document(source_name: str):
    """Remove a document and all its chunks from the vector store."""
    removed = get_store().delete_by_source(source_name)
    if removed == 0:
        return jsonify({"error": f"No chunks found for '{source_name}'"}), 404
    return jsonify({"message": f"Deleted {removed} chunks for '{source_name}'"})
 
 
@api.route("/metadata/values/<field>")
def metadata_values(field: str):
    """Returns all distinct values for a metadata field (used by UI dropdowns)."""
    vals = get_store().list_metadata_values(field)
    return jsonify({"field": field, "values": sorted(str(v) for v in vals if v)})
 
 
# ── Error Handlers ─────────────────────────────────────────────────────────────
 
def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404
 
    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed"}), 405
 
    @app.errorhandler(429)
    def rate_limited(e):
        return jsonify({"error": "Rate limit exceeded. Please wait a moment."}), 429
 
    @app.errorhandler(500)
    def internal_error(e):
        logger.error("unhandled_exception", error=str(e))
        return jsonify({"error": "Internal server error"}), 500
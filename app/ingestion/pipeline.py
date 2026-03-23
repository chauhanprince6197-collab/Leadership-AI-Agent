"""
app/ingestion/pipeline.py — Document Ingestion Pipeline

Uses:
  - LangChain document loaders  (PDF, DOCX, TXT, HTML, PPTX)
  - LangChain RecursiveCharacterTextSplitter
  - Rich metadata extraction per chunk
"""

from __future__ import annotations
import re
import hashlib
import logging
from pathlib import Path
from typing import Optional

import structlog

# ── LangChain document loaders ────────────────────────────────────────────
from langchain_community.document_loaders import (
    PyPDFLoader,           # PDF → per-page Documents
    Docx2txtLoader,        # .docx
    TextLoader,            # .txt / .md
    UnstructuredHTMLLoader,# .html
    UnstructuredPowerPointLoader,  # .pptx
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


# ── Document type classifier ──────────────────────────────────────────────

DOC_TYPE_PATTERNS: dict[str, list[str]] = {
    "annual_report":    [r"annual report", r"annual_report", r"fy\d{4}"],
    "quarterly_report": [r"quarterly report", r"q[1-4][\s_]\d{4}", r"q[1-4]_?\d{4}"],
    "strategy":         [r"strateg", r"roadmap", r"\d[-\s]year plan", r"vision"],
    "operational":      [r"operational", r"operations update", r"ops"],
    "financial":        [r"financ", r"budget", r"p&l", r"balance sheet"],
    "hr":               [r"human resources", r"talent", r"headcount", r"attrition"],
}

HEADING_RE = re.compile(
    r"^(?:#{1,4}\s+.+|[A-Z][A-Z0-9 &/\-]{3,}(?:\s*[-=]{3,})?)$",
    re.MULTILINE,
)


def _infer_doc_type(source: str, preview: str) -> str:
    combined = (source + " " + preview[:400]).lower()
    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        if any(re.search(p, combined) for p in patterns):
            return doc_type
    return "general"


def _nearest_section(text: str, full_doc_text: str, approx_offset: int) -> str:
    """Return the heading that most recently precedes this chunk position."""
    preceding = full_doc_text[:approx_offset]
    headings = HEADING_RE.findall(preceding)
    return headings[-1].strip("-= \n") if headings else "General"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ── Loader registry ───────────────────────────────────────────────────────

LOADER_MAP: dict[str, type] = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
    ".html": UnstructuredHTMLLoader,
    ".pptx": UnstructuredPowerPointLoader,
}


def load_file(path: Path) -> list[Document]:
    """
    Load a single file using the appropriate LangChain loader.
    Returns a list of LangChain Document objects (one per page for PDFs).
    Raises ValueError for unsupported types.
    """
    ext = path.suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")

    try:
        loader = loader_cls(str(path))
        docs = loader.load()
        logger.info("file_loaded", file=path.name, pages=len(docs))
        return docs
    except Exception as exc:
        logger.error("file_load_failed", file=path.name, error=str(exc))
        raise


# ── Splitter factory ──────────────────────────────────────────────────────

def make_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter with a document-structure-aware separator
    hierarchy: sections → paragraphs → sentences → words → characters.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",   # major section break
            "\n\n",     # paragraph break
            "\n",       # line break
            ". ",       # sentence end
            ", ",       # clause
            " ",        # word
            "",         # character fallback
        ],
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,
        add_start_index=True,   # adds "start_index" to metadata — crucial for section lookup
    )


# ── Main pipeline ─────────────────────────────────────────────────────────

def ingest_file(
    path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Full pipeline for a single file:
      load → split → enrich metadata → return chunk dicts

    Returns list of:
    {
        "chunk_id":  str,   # deterministic: sha256(source+index)
        "text":      str,
        "metadata": {
            "source":        str,   # filename
            "doc_type":      str,   # inferred category
            "chunk_index":   int,
            "total_chunks":  int,
            "char_count":    int,
            "word_count":    int,
            "has_numbers":   bool,
            "section":       str,   # nearest heading
            "content_hash":  str,   # deduplication key
            "file_path":     str,   # absolute path (for re-ingestion)
        }
    }
    """
    raw_docs = load_file(path)
    splitter = make_splitter(chunk_size, chunk_overlap)

    # Collect full doc text for section extraction
    full_text = "\n\n".join(d.page_content for d in raw_docs)
    doc_type  = _infer_doc_type(path.name, full_text)

    # Add source metadata before splitting (LangChain propagates it)
    for doc in raw_docs:
        doc.metadata.setdefault("source", path.name)

    # Split — LangChain adds start_index to metadata when add_start_index=True
    split_docs: list[Document] = splitter.split_documents(raw_docs)
    total = len(split_docs)

    chunks = []
    for i, doc in enumerate(split_docs):
        text = doc.page_content.strip()
        if not text:
            continue

        start_idx = doc.metadata.get("start_index", 0)
        section   = _nearest_section(text, full_text, start_idx)

        chunks.append({
            "chunk_id": f"{path.stem}_{i:04d}",
            "text": text,
            "metadata": {
                "source":       path.name,
                "doc_type":     doc_type,
                "chunk_index":  i,
                "total_chunks": total,
                "char_count":   len(text),
                "word_count":   len(text.split()),
                "has_numbers":  bool(re.search(r"\$[\d,.]+|\d+[%BMK]|\d+\.\d+", text)),
                "section":      section,
                "content_hash": _content_hash(text),
                "file_path":    str(path.resolve()),
            },
        })

    logger.info("file_ingested",
                file=path.name, doc_type=doc_type,
                chunks=len(chunks), total_chars=sum(c["metadata"]["char_count"] for c in chunks))
    return chunks


def ingest_folder(
    folder: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    glob_pattern: str = "**/*",
) -> list[dict]:
    """
    Recursively ingest all supported files in a folder.
    Skips unsupported types with a warning. Logs failures per-file without
    crashing the whole ingestion.
    """
    all_chunks: list[dict] = []
    supported = set(LOADER_MAP.keys())

    files = [p for p in folder.glob(glob_pattern)
             if p.is_file() and p.suffix.lower() in supported]

    if not files:
        logger.warning("no_files_found", folder=str(folder))
        return []

    for fp in files:
        try:
            chunks = ingest_file(fp, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.error("file_skip", file=fp.name, reason=str(exc))
            continue

    logger.info("folder_ingested",
                folder=str(folder),
                files=len(files),
                total_chunks=len(all_chunks))
    return all_chunks

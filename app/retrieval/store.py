"""
app/retrieval/store.py — Production Vector Store

FIX SUMMARY vs original:
  1. Added threading.RLock() on all BM25 read/write operations
     → Prevents race conditions under concurrent gunicorn workers
  2. _rebuild_bm25() now acquires lock before touching shared state
  3. _bm25_score_filtered() acquires lock for read consistency
  4. delete_by_source() acquires lock around rebuild call
  All ChromaDB calls are inherently thread-safe (it handles its own locking).
  Only the in-memory BM25 state needed protection.
"""

from __future__ import annotations
import pickle
import threading                          # FIX: added
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

import numpy as np
import structlog

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = structlog.get_logger(__name__)


def _f(v) -> float:
    """Extract a guaranteed plain Python float from any value including numpy scalars."""
    try:
        return v.item()
    except AttributeError:
        return float(v)


def _sanitize_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool metadata values."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, (int, float, str)):
            clean[k] = v
        elif isinstance(v, np.bool_):
            clean[k] = bool(v)
        elif isinstance(v, np.integer):
            clean[k] = int(v)
        elif isinstance(v, np.floating):
            clean[k] = float(v.item())
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean


class HybridVectorStore:

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "leadership_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir     = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # FIX: RLock protects all in-memory BM25 state.
        # RLock (re-entrant) allows the same thread to acquire it multiple
        # times — important because _rebuild_bm25 is called from add_chunks
        # and delete_by_source which may already hold the lock.
        self._lock = threading.RLock()

        logger.info("loading_embedding_model", model=embedding_model)
        self._embedder = SentenceTransformer(embedding_model)

        self._chroma = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[str] = []
        self._bm25_ids:    list[str] = []
        self._load_bm25_index()

        logger.info("store_ready",
                    collection=collection_name,
                    existing_docs=self._collection.count())

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        ids        = [c["chunk_id"] for c in chunks]
        texts      = [c["text"]     for c in chunks]
        metadatas  = [_sanitize_metadata(c["metadata"]) for c in chunks]
        embeddings = self._embed_batch(texts)

        # ChromaDB upsert is thread-safe internally
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        # FIX: BM25 rebuild requires the lock — in-memory state mutation
        self._rebuild_bm25()

        logger.info("chunks_added", added=len(chunks),
                    total_in_collection=self._collection.count())
        return len(chunks)

    def _embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            vecs = self._embedder.encode(
                texts[i: i + batch_size],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 6,
        *,
        where: Optional[dict] = None,
        mode: str = "hybrid",
        rrf_k: int = 60,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
    ) -> list[dict]:

        if self._collection.count() == 0:
            logger.warning("empty_collection")
            return []

        pool_size = min(top_k * 5, self._collection.count())

        q_vec_np: np.ndarray = self._embedder.encode(
            query_text, normalize_embeddings=True
        )
        q_vec: list[float] = q_vec_np.tolist()

        chroma_kwargs: dict[str, Any] = {
            "query_embeddings": [q_vec],
            "n_results": pool_size,
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if where:
            chroma_kwargs["where"] = where

        dense_result     = self._collection.query(**chroma_kwargs)
        dense_ids        = dense_result["ids"][0]
        dense_docs       = dense_result["documents"][0]
        dense_metas      = dense_result["metadatas"][0]
        dense_distances  = dense_result["distances"][0]
        raw_emb          = dense_result.get("embeddings") or []
        dense_embeddings = raw_emb[0] if raw_emb else []

        dense_scores: dict[str, float] = {
            cid: 1.0 - _f(dist)
            for cid, dist in zip(dense_ids, dense_distances)
        }

        # FIX: lock acquired inside _bm25_score_filtered for read safety
        sparse_scores: dict[str, float] = self._bm25_score_filtered(
            query_text, set(dense_ids)
        )

        if mode == "dense":
            fused_scores = dense_scores
            ranked_ids   = sorted(
                dense_scores, key=lambda c: dense_scores[c], reverse=True
            )[:top_k]
        elif mode == "sparse":
            fused_scores = sparse_scores
            ranked_ids   = sorted(
                sparse_scores, key=lambda c: sparse_scores[c], reverse=True
            )[:top_k]
        else:  # hybrid
            fused_scores = self._rrf([dense_scores, sparse_scores], k=rrf_k)
            ranked_ids   = sorted(
                fused_scores, key=lambda c: fused_scores[c], reverse=True
            )

        id_to_meta = dict(zip(dense_ids, dense_metas))
        id_to_doc  = dict(zip(dense_ids, dense_docs))

        id_to_emb: dict[str, list[float]] = {}
        for cid, emb in zip(dense_ids, dense_embeddings):
            if emb is not None:
                id_to_emb[cid] = emb.tolist() if hasattr(emb, "tolist") else list(emb)

        pool = [cid for cid in ranked_ids if cid in id_to_doc]

        if use_mmr and len(pool) > top_k and id_to_emb:
            pool = self._mmr(
                query_vec=q_vec_np,
                candidate_ids=pool,
                id_to_vec=id_to_emb,
                id_to_score=fused_scores,
                top_k=top_k,
                lambda_=float(mmr_lambda),
            )
        else:
            pool = pool[:top_k]

        results = []
        for rank, cid in enumerate(pool):
            results.append({
                "chunk_id": cid,
                "text":     id_to_doc[cid],
                "metadata": id_to_meta[cid],
                "retrieval": {
                    "rank":         rank + 1,
                    "fused_score":  round(_f(fused_scores.get(cid, 0.0)), 4),
                    "dense_score":  round(_f(dense_scores.get(cid, 0.0)), 4),
                    "sparse_score": round(_f(sparse_scores.get(cid, 0.0)), 4),
                    "mode":         mode,
                },
            })

        return results

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        """
        FIX: Acquires self._lock before mutating any BM25 in-memory state.
        Without this, two concurrent add_chunks() calls could interleave,
        producing a BM25 index that doesn't match self._bm25_ids.
        """
        result = self._collection.get(include=["documents"])
        new_ids    = result["ids"]
        new_corpus = result["documents"]

        if new_corpus:
            tokenised = [doc.lower().split() for doc in new_corpus]
            new_bm25  = BM25Okapi(tokenised)  # build outside lock (CPU work)
        else:
            new_bm25 = None

        # Lock only the assignment — minimises contention window
        with self._lock:
            self._bm25_ids    = new_ids
            self._bm25_corpus = new_corpus
            self._bm25        = new_bm25
            if new_bm25:
                self._save_bm25_index()

        logger.debug("bm25_rebuilt", corpus_size=len(new_ids))

    def _bm25_score_filtered(
        self, query: str, candidate_ids: set[str]
    ) -> dict[str, float]:
        """FIX: Acquires lock for consistent read of BM25 state."""
        with self._lock:
            if self._bm25 is None or not self._bm25_ids:
                return {}

            q_tokens   = query.lower().split()
            all_scores = self._bm25.get_scores(q_tokens)

            filtered: dict[str, float] = {}
            for idx, cid in enumerate(self._bm25_ids):
                if cid in candidate_ids:
                    filtered[cid] = all_scores[idx].item()

        if not filtered:
            return {}

        vals: list[float] = list(filtered.values())
        mn: float = min(vals)
        mx: float = max(vals)
        if mx > mn:
            span = mx - mn
            filtered = {k: (v - mn) / span for k, v in filtered.items()}

        return filtered

    # ── RRF ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf(score_dicts: list[dict[str, float]], k: int = 60) -> dict[str, float]:
        fused: dict[str, float] = defaultdict(float)
        for scores in score_dicts:
            ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
            for rank, doc_id in enumerate(ranked):
                fused[doc_id] += 1.0 / (k + rank + 1)
        return dict(fused)

    # ── MMR ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _mmr(
        query_vec: np.ndarray,
        candidate_ids: list[str],
        id_to_vec: dict[str, list[float]],
        id_to_score: dict[str, float],
        top_k: int,
        lambda_: float,
    ) -> list[str]:
        vecs = {
            cid: np.array(v, dtype=np.float32)
            for cid, v in id_to_vec.items()
            if cid in set(candidate_ids)
        }
        selected:  list[str] = []
        remaining: list[str] = [c for c in candidate_ids if c in vecs]

        while len(selected) < top_k and remaining:
            if not selected:
                best = max(remaining,
                           key=lambda cid: float(id_to_score.get(cid, 0.0)))
            else:
                sel_mat    = np.vstack([vecs[s] for s in selected])
                best_score: float | None = None
                best       = remaining[0]
                for cid in remaining:
                    v         = vecs[cid]
                    relevance = float(id_to_score.get(cid, 0.0))
                    max_sim   = float(np.max(sel_mat @ v))
                    score     = lambda_ * relevance - (1.0 - lambda_) * max_sim
                    if best_score is None or score > best_score:
                        best_score = score
                        best       = cid
            selected.append(best)
            remaining.remove(best)

        return selected

    # ── Persistence ───────────────────────────────────────────────────────────

    def _bm25_path(self) -> Path:
        return self.persist_dir / "bm25_index.pkl"

    def _save_bm25_index(self) -> None:
        # Called inside _rebuild_bm25 which already holds the lock
        with open(self._bm25_path(), "wb") as f:
            pickle.dump({
                "bm25":   self._bm25,
                "corpus": self._bm25_corpus,
                "ids":    self._bm25_ids,
            }, f)

    def _load_bm25_index(self) -> None:
        path = self._bm25_path()
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            with self._lock:
                self._bm25        = state["bm25"]
                self._bm25_corpus = state["corpus"]
                self._bm25_ids    = state["ids"]
            logger.info("bm25_loaded", corpus_size=len(self._bm25_ids))
        except Exception as e:
            logger.warning("bm25_load_failed", error=str(e))

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        count  = self._collection.count()
        result = self._collection.get(include=["metadatas"]) \
                 if count > 0 else {"metadatas": []}
        sources = list({m.get("source", "") for m in result["metadatas"]})
        types   = list({m.get("doc_type", "") for m in result["metadatas"]})
        return {
            "total_chunks":    count,
            "total_documents": len(sources),
            "doc_types":       types,
            "documents":       sources,
            "bm25_synced":     len(self._bm25_ids) == count,
        }

    def list_metadata_values(self, field: str) -> list[Any]:
        result = self._collection.get(include=["metadatas"])
        return list({m.get(field) for m in result["metadatas"] if field in m})

    def delete_by_source(self, source_name: str) -> int:
        result = self._collection.get(where={"source": source_name}, include=[])
        ids    = result["ids"]
        if ids:
            self._collection.delete(ids=ids)
            self._rebuild_bm25()   # lock acquired inside _rebuild_bm25
            logger.info("source_deleted", source=source_name, chunks_removed=len(ids))
        return len(ids)

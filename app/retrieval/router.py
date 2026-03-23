"""
app/retrieval/router.py — Query Router

Analyses the natural-language question and returns optimal retrieval params.
Centralising this logic makes it easy to test and extend.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalParams:
    mode:       str            # "dense" | "sparse" | "hybrid"
    where:      Optional[dict] # ChromaDB metadata filter
    mmr_lambda: float          # 0=diversity, 1=pure relevance
    top_k:      int


# Keyword → (doc_type_filter, mode, mmr_lambda)
_ROUTING_RULES: list[tuple[list[str], dict, str, float]] = [
    (
        ["revenue", "profit", "growth", "margin", "cost", "budget",
         "earnings", "ebitda", "cash", "financial", "sales figure"],
        {"has_numbers": True},
        "hybrid", 0.7,     # precision > diversity for financial data
    ),
    (
        ["risk", "threat", "challenge", "issue", "concern", "problem",
         "gap", "vulnerability", "exposure"],
        {"doc_type": {"$in": ["quarterly_report", "annual_report",
                               "operational", "strategy"]}},
        "hybrid", 0.45,    # want diversity — risks span docs
    ),
    (
        ["last quarter", "q3", "q2", "q1", "q4", "quarterly",
         "recent", "latest", "this quarter", "month", "november",
         "october", "september"],
        {"doc_type": {"$in": ["quarterly_report", "operational"]}},
        "hybrid", 0.6,
    ),
    (
        ["strategy", "plan", "roadmap", "vision", "2025", "2026",
         "2027", "future", "goal", "objective", "pillar", "initiative"],
        {"doc_type": {"$in": ["strategy", "annual_report"]}},
        "dense", 0.45,     # semantic similarity beats keyword for strategy
    ),
    (
        ["department", "team", "hr", "employee", "headcount", "attrition",
         "talent", "retention", "engineering", "sales", "marketing",
         "operations", "underperform", "hiring", "glassdoor"],
        {"doc_type": {"$in": ["quarterly_report", "operational"]}},
        "hybrid", 0.5,
    ),
    (
        ["acquisition", "m&a", "merger", "datasync", "acqui"],
        {"doc_type": {"$in": ["annual_report", "quarterly_report", "strategy"]}},
        "hybrid", 0.5,
    ),
]


def route(question: str, default_top_k: int = 6) -> RetrievalParams:
    """
    Match question against routing rules.
    Returns the first matching rule; falls back to a sensible default.
    """
    q = question.lower()

    for keywords, where, mode, mmr_lambda in _ROUTING_RULES:
        if any(kw in q for kw in keywords):
            return RetrievalParams(
                mode=mode, where=where,
                mmr_lambda=mmr_lambda, top_k=default_top_k
            )

    # No rule matched — full corpus, hybrid retrieval
    return RetrievalParams(
        mode="hybrid", where=None,
        mmr_lambda=0.5, top_k=default_top_k
    )

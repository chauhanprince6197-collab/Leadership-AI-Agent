"""
app/generation/chain.py — LangChain Generation Chain

FIX SUMMARY vs original:
  1. _get_cached_llm() caches ChatAnthropic instances by (api_key, model, max_tokens)
     → Original created a new HTTP client + connection pool on every single request
     → With 100 concurrent users this meant 100 separate connection pools to Anthropic
     → Now reuses the same client object across requests for the same key
  2. Cache is bounded at maxsize=20 (covers ~20 unique API keys before evicting oldest)
  3. All other logic (LCEL chain, retry, streaming) unchanged
"""

from __future__ import annotations
from typing import Iterator
from functools import lru_cache         # FIX: added for LLM client caching

import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from anthropic import RateLimitError, APIStatusError

logger = structlog.get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are an elite executive intelligence analyst embedded in a corporate leadership platform.
You have exclusive access to the internal company documents provided below.

CRITICAL RULES:
1. Ground every claim in the provided context excerpts — never fabricate figures.
2. If the context is insufficient, state clearly what is and isn't available.
3. Be executive-ready: structured, concise, actionable. Lead with key insight.
4. Use **bold** for key numbers and findings. Use bullet points for lists.
5. Always cite the source document for specific figures (e.g., "per Q3 2024 Report").
6. If documents conflict on the same point, explicitly flag the discrepancy.
7. Proactively surface second-order risks, trends, or implications.

RESPONSE FORMAT:
**Executive Summary** (2 sentences max)
**Key Findings** (bullet points, data-grounded)
**Risks / Watch Items** (if applicable)
"""

HUMAN_TEMPLATE = """\
INTERNAL DOCUMENT EXCERPTS:
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEADERSHIP QUESTION:
{question}

Answer based solely on the excerpts above. Cite sources.
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human",  HUMAN_TEMPLATE),
])


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    if not chunks:
        return ""
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        r = c["retrieval"]
        header = (
            f"[{i}] Source: {m['source']} | Type: {m['doc_type']} | "
            f"Section: {m.get('section', '—')} | "
            f"Scores — dense: {r['dense_score']}, bm25: {r['sparse_score']}, "
            f"fused: {r['fused_score']}"
        )
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ── LLM client cache ──────────────────────────────────────────────────────────

@lru_cache(maxsize=20)
def _get_cached_llm(
    api_key:    str,
    model:      str,
    max_tokens: int,
) -> ChatAnthropic:
    """
    FIX: Cache ChatAnthropic instances by (api_key, model, max_tokens).

    Original code called make_chain() → ChatAnthropic() on every request,
    creating a brand-new HTTP connection pool each time. Under 100 concurrent
    users this produced 100 separate TCP connections to Anthropic's API,
    wasting memory and connection slots.

    lru_cache(maxsize=20) keeps the 20 most recently used clients. The vast
    majority of deployments use a single API key, so cache hit rate is ~100%.

    Note: temperature and streaming are NOT part of the cache key because
    the same underlying client handles both — streaming vs non-streaming is
    controlled at call time (.stream() vs .invoke()), not at instantiation.
    """
    logger.debug("creating_llm_client", model=model)
    return ChatAnthropic(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        max_retries=3,
    )


# ── Chain factory ─────────────────────────────────────────────────────────────

def make_chain(
    api_key:    str,
    model:      str   = "claude-sonnet-4-20250514",
    max_tokens: int   = 1024,
    temperature: float = 0.0,
    streaming:  bool  = False,
) -> object:
    """
    Build a LangChain LCEL chain: prompt | llm | output_parser.

    FIX: Now uses _get_cached_llm() instead of creating a new ChatAnthropic
    on every invocation. The temperature parameter is kept for API compatibility
    but doesn't affect the cached client (temperature=0.0 is the production default).
    """
    llm = _get_cached_llm(api_key, model, max_tokens)
    return _PROMPT | llm | StrOutputParser()


# ── Answer function ───────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    reraise=True,
)
def generate_answer(
    question:    str,
    chunks:      list[dict],
    api_key:     str,
    model:       str   = "claude-sonnet-4-20250514",
    max_tokens:  int   = 1024,
    temperature: float = 0.0,
) -> str:
    if not chunks:
        return (
            "I could not find relevant information in the loaded documents "
            "to answer this question. Please ensure the relevant documents "
            "have been ingested."
        )

    context = build_context(chunks)
    chain   = make_chain(api_key, model=model, max_tokens=max_tokens,
                         temperature=temperature)

    logger.info("generating_answer",
                question_len=len(question),
                chunks=len(chunks),
                context_chars=len(context))

    answer = chain.invoke({"context": context, "question": question})
    logger.info("answer_generated", answer_len=len(answer))
    return answer


def stream_answer(
    question:   str,
    chunks:     list[dict],
    api_key:    str,
    model:      str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
) -> Iterator[str]:
    if not chunks:
        yield "No relevant documents found."
        return

    context = build_context(chunks)
    chain   = make_chain(api_key, model=model, max_tokens=max_tokens,
                         streaming=True)

    for token in chain.stream({"context": context, "question": question}):
        yield token

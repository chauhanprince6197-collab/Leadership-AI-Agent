"""
app/generation/chain.py — LangChain Generation Chain (OpenAI version)
"""

from __future__ import annotations
from typing import Iterator

import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = structlog.get_logger(__name__)

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


def make_chain(
    api_key: str,
    model: str        = "gpt-4o",
    max_tokens: int   = 1024,
    temperature: float = 0.0,
    streaming: bool   = False,
) -> object:
    """
    Build a LangChain LCEL chain using OpenAI:  prompt | llm | output_parser
    """
    llm = ChatOpenAI(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        streaming=streaming,
        max_retries=3,
    )
    return _PROMPT | llm | StrOutputParser()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def generate_answer(
    question:    str,
    chunks:      list[dict],
    api_key:     str,
    model:       str   = "gpt-4o",
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
                question_len=len(question), chunks=len(chunks),
                context_chars=len(context))

    answer = chain.invoke({"context": context, "question": question})
    logger.info("answer_generated", answer_len=len(answer))
    return answer


def stream_answer(
    question:   str,
    chunks:     list[dict],
    api_key:    str,
    model:      str = "gpt-4o",
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

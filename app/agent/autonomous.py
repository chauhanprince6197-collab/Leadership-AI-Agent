"""
app/agent/autonomous.py — Task 2: Autonomous Decision Agent

Evolves the Leadership Insight Agent into a multi-step autonomous reasoning
system that handles open-ended strategic questions by:

  Step 1 — PLAN     : Claude decomposes the question into 3-5 sub-questions
  Step 2 — RESEARCH : RAG pipeline retrieves evidence per sub-question
  Step 3 — SYNTHESISE: Claude produces a structured executive decision brief

Usage:
    from app.agent.autonomous import AutonomousDecisionAgent

    agent = AutonomousDecisionAgent(store=store, api_key="sk-ant-...")
    result = agent.run("Should we accelerate APAC expansion in 2025?")
    print(result.decision_brief)

    # Or stream token-by-token:
    for token in agent.stream("Should we accelerate APAC expansion in 2025?"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterator

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.retrieval.store import HybridVectorStore
from app.retrieval.router import route
from app.generation.chain import build_context

logger = structlog.get_logger(__name__)


# ── Prompt Templates ──────────────────────────────────────────────────────────

_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a strategic research planner for a corporate leadership team.
Your job is to decompose a complex strategic question into 3-5 focused
sub-questions that, when answered together, fully address the original question.

Rules:
- Each sub-question must be self-contained and answerable from company documents
- Order them from foundational (current state) to forward-looking (implications)
- Return ONLY a valid JSON array of strings — no explanation, no markdown, no extra text

Example output:
["What is the current APAC revenue and growth rate?", \
"What does the 3-year strategy say about APAC targets?", \
"What operational risks exist in our APAC markets?", \
"What is our available cash to fund expansion?"]
"""),
    ("human", "Strategic question to decompose: {question}"),
])

_RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a precise corporate analyst. Answer the specific research question below
using ONLY the provided document excerpts. Be concise and factual.
Cite the source document name when referencing specific figures.
If the context does not contain enough information, say so explicitly.
"""),
    ("human", """\
DOCUMENT EXCERPTS:
{context}

RESEARCH QUESTION: {sub_question}

Provide a concise factual answer based solely on the excerpts above.
"""),
])

_SYNTHESISER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an elite management consultant preparing an executive decision brief for a CEO.
You have completed multi-document research across several sub-questions.
Synthesise all findings into a structured, actionable decision brief.

Use EXACTLY this format:

## Executive Summary
(2-3 sentences: core situation and recommended direction)

## Key Findings
(Bullet points — each grounded in research, cite source documents)

## Strategic Options
### Option 1: [Name]
- Pros: ...
- Cons: ...

### Option 2: [Name]
- Pros: ...
- Cons: ...

### Option 3: [Name] (if applicable)
- Pros: ...
- Cons: ...

## Recommended Action
(Clear recommendation with rationale and suggested timeline)

## Key Risks to Monitor
(Top 3 risks with suggested mitigation for each)

## Data Gaps
(What additional information would strengthen this analysis)
"""),
    ("human", """\
ORIGINAL STRATEGIC QUESTION:
{question}

RESEARCH FINDINGS:
{research_summary}

Produce the executive decision brief now.
"""),
])


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ResearchStep:
    """Result of researching one sub-question."""
    sub_question: str
    chunks:       list[dict]
    answer:       str
    sources:      list[str]
    duration_s:   float


@dataclass
class AgentResult:
    """Complete result from the autonomous agent pipeline."""
    question:       str
    plan:           list[str]
    research_steps: list[ResearchStep]
    decision_brief: str
    total_duration: float
    total_chunks:   int
    all_sources:    list[str]


# ── Agent Class ───────────────────────────────────────────────────────────────

class AutonomousDecisionAgent:
    """
    Multi-step autonomous agent: Plan → Research → Synthesise → Decision Brief.

    Args:
        store:      HybridVectorStore instance (already loaded with documents)
        api_key:    Anthropic API key (sk-ant-...)
        model:      Claude model name
        max_tokens: Max tokens for generation
        top_k:      Number of chunks to retrieve per sub-question
    """

    def __init__(
        self,
        store:      HybridVectorStore,
        api_key:    str,
        model:      str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
        top_k:      int = 5,
    ):
        self.store      = store
        self.api_key    = api_key
        self.model      = model
        self.max_tokens = max_tokens
        self.top_k      = top_k

        # Build the LangChain LLM once — shared across all three chains
        _llm = ChatAnthropic(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            max_retries=3,
        )
        _parser = StrOutputParser()

        # Three LCEL chains: planner, researcher, synthesiser
        self._planner     = _PLANNER_PROMPT     | _llm | _parser
        self._researcher  = _RESEARCHER_PROMPT  | _llm | _parser
        self._synthesiser = _SYNTHESISER_PROMPT | _llm | _parser

    # ── Public: Standard (blocking) run ──────────────────────────────────────

    def run(self, question: str) -> AgentResult:
        """
        Run the full autonomous pipeline and return a complete AgentResult.
        Blocks until all steps are done.
        """
        t0 = time.time()
        logger.info("agent_start", question=question[:80])

        # Step 1: Plan
        plan = self._plan(question)
        logger.info("agent_planned", sub_questions=len(plan), plan=plan)

        # Step 2: Research each sub-question independently
        research_steps: list[ResearchStep] = []
        for i, sub_q in enumerate(plan):
            logger.info("agent_researching",
                        step=i + 1, total=len(plan),
                        sub_question=sub_q[:70])
            step = self._research_one(sub_q)
            research_steps.append(step)

        # Step 3: Synthesise all findings into a decision brief
        research_summary = self._format_research_summary(research_steps)
        decision_brief   = self._synthesiser.invoke({
            "question":         question,
            "research_summary": research_summary,
        })

        # Aggregate metadata
        all_sources  = list({s for step in research_steps for s in step.sources})
        total_chunks = sum(len(s.chunks) for s in research_steps)
        elapsed      = round(time.time() - t0, 1)

        logger.info("agent_complete",
                    duration_s=elapsed,
                    total_chunks=total_chunks,
                    unique_sources=len(all_sources))

        return AgentResult(
            question       = question,
            plan           = plan,
            research_steps = research_steps,
            decision_brief = decision_brief,
            total_duration = elapsed,
            total_chunks   = total_chunks,
            all_sources    = all_sources,
        )

    # ── Public: Streaming run ─────────────────────────────────────────────────

    def stream(self, question: str) -> Iterator[str]:
        """
        Run the full pipeline and stream output token-by-token.
        Yields progress updates as the agent works, then streams the final brief.
        Use with Flask's stream_with_context().
        """
        t0 = time.time()

        # Stream: planning step
        yield "**Step 1 — Planning research approach...**\n\n"
        plan = self._plan(question)
        yield f"Identified **{len(plan)} research areas:**\n\n"
        for i, q in enumerate(plan, 1):
            yield f"**{i}.** {q}\n\n"
        yield "---\n\n"

        # Stream: research steps
        research_steps: list[ResearchStep] = []
        for i, sub_q in enumerate(plan):
            yield f"**Step 2.{i+1} — Researching:** {sub_q}\n\n"
            step = self._research_one(sub_q)
            research_steps.append(step)
            src_str = ", ".join(f"`{s}`" for s in step.sources) if step.sources else "no matching documents"
            yield f"*Retrieved {len(step.chunks)} passages from: {src_str}*\n\n"

        # Stream: synthesis
        yield "---\n\n**Step 3 — Synthesising findings into decision brief...**\n\n"

        research_summary = self._format_research_summary(research_steps)
        for token in self._synthesiser.stream({
            "question":         question,
            "research_summary": research_summary,
        }):
            yield token

        # Stream: footer
        elapsed      = round(time.time() - t0, 1)
        total_chunks = sum(len(s.chunks) for s in research_steps)
        all_sources  = list({s for step in research_steps for s in step.sources})
        yield (
            f"\n\n---\n"
            f"*Analysis complete in **{elapsed}s** · "
            f"**{total_chunks}** passages reviewed · "
            f"Sources: {', '.join(f'`{s}`' for s in all_sources) or 'none'}*"
        )

    # ── Private: Plan ─────────────────────────────────────────────────────────

    def _plan(self, question: str) -> list[str]:
        """Ask Claude to decompose the question into focused sub-questions."""
        raw = self._planner.invoke({"question": question})

        # Parse JSON array — strip markdown fences if Claude added them
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                # Remove ```json ... ``` or ``` ... ```
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()
            plan = json.loads(clean)
            if isinstance(plan, list) and all(isinstance(q, str) for q in plan):
                return plan[:5]   # cap at 5 sub-questions
        except (json.JSONDecodeError, IndexError, ValueError):
            pass

        # Fallback: split on newlines if JSON parsing fails
        lines = [
            line.strip().lstrip("0123456789.-) ")
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 10
        ]
        return lines[:5] if lines else [question]

    # ── Private: Research one sub-question ───────────────────────────────────

    def _research_one(self, sub_question: str) -> ResearchStep:
        """Retrieve evidence and generate an answer for a single sub-question."""
        t0 = time.time()

        # Use the query router for smart filtering per sub-question type
        params = route(sub_question, default_top_k=self.top_k)
        chunks = self.store.query(
            query_text=sub_question,
            top_k=self.top_k,
            where=params.where,
            mode=params.mode,
            mmr_lambda=params.mmr_lambda,
        )

        if chunks:
            context = build_context(chunks)
            answer  = self._researcher.invoke({
                "context":      context,
                "sub_question": sub_question,
            })
            sources = list({c["metadata"]["source"] for c in chunks})
        else:
            answer  = (
                "No relevant information found in the loaded documents "
                "for this specific question."
            )
            sources = []

        return ResearchStep(
            sub_question = sub_question,
            chunks       = chunks,
            answer       = answer,
            sources      = sources,
            duration_s   = round(time.time() - t0, 2),
        )

    # ── Private: Format research summary ─────────────────────────────────────

    def _format_research_summary(self, steps: list[ResearchStep]) -> str:
        """Format all research steps into a readable summary for the synthesiser."""
        parts = []
        for i, step in enumerate(steps, 1):
            src_str = ", ".join(step.sources) if step.sources else "no documents matched"
            parts.append(
                f"### Research Area {i}: {step.sub_question}\n"
                f"**Sources consulted:** {src_str}\n"
                f"**Duration:** {step.duration_s}s\n\n"
                f"{step.answer}"
            )
        return "\n\n---\n\n".join(parts)

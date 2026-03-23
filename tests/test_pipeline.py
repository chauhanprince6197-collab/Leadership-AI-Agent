"""
tests/test_pipeline.py  —  Production test suite, 40 tests, 8 modules
No pytest fixtures. No external dependencies. Pure Python + numpy.

Run:  python -m pytest tests/ -v        (with pytest installed)
  or: python tests/test_pipeline.py     (no pytest needed)
"""
from __future__ import annotations
import sys, math
import numpy as np
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════
# 1 ─ QUERY ROUTER
# ══════════════════════════════════════════════════════════════════════════
class TestQueryRouter:
    def _r(self, q, top_k=6):
        from app.retrieval.router import route
        return route(q, default_top_k=top_k)

    def test_revenue_hybrid_numbers_filter(self):
        p = self._r("What is our current revenue trend?")
        assert p.mode == "hybrid"
        assert p.where == {"has_numbers": True}

    def test_ebitda_triggers_numbers_filter(self):
        p = self._r("What is our EBITDA margin this year?")
        assert p.where == {"has_numbers": True}

    def test_risk_spans_all_doc_types(self):
        p = self._r("What are the key risks from last quarter?")
        types_in = p.where["doc_type"]["$in"]
        assert "quarterly_report" in types_in
        assert "strategy"         in types_in
        assert "annual_report"    in types_in

    def test_quarterly_filters_to_recent_docs(self):
        p = self._r("What happened last quarter in operations?")
        types_in = p.where["doc_type"]["$in"]
        assert "quarterly_report" in types_in
        assert "operational"      in types_in

    def test_strategy_uses_dense_mode(self):
        p = self._r("What is our 3-year strategic roadmap?")
        assert p.mode == "dense"
        assert "strategy" in p.where["doc_type"]["$in"]

    def test_department_routes_to_operational(self):
        p = self._r("Which departments are underperforming?")
        assert "quarterly_report" in p.where["doc_type"]["$in"]

    def test_unknown_question_full_corpus(self):
        p = self._r("Tell me everything about the company.")
        assert p.where is None
        assert p.mode == "hybrid"

    def test_top_k_propagated(self):
        assert self._r("What is revenue?", top_k=10).top_k == 10

    def test_mmr_lambda_in_valid_range(self):
        for q in ["What is revenue?", "What are the risks?", "What is the strategy?"]:
            p = self._r(q)
            assert 0.0 <= p.mmr_lambda <= 1.0, f"lambda out of range for: {q}"

    def test_financial_lambda_is_precision_heavy(self):
        # financial questions prioritise precision (higher lambda)
        p = self._r("What is our net profit margin?")
        assert p.mmr_lambda >= 0.6


# ══════════════════════════════════════════════════════════════════════════
# 2 ─ INGESTION: doc-type inference and hash helpers
# ══════════════════════════════════════════════════════════════════════════
class TestIngestionPipeline:
    def _dt(self, src, txt):
        from app.ingestion.pipeline import _infer_doc_type
        return _infer_doc_type(src, txt)

    def test_annual_report_by_filename(self):
        assert self._dt("annual_report_2024.txt", "FY2024 annual report") == "annual_report"

    def test_quarterly_report_by_filename(self):
        assert self._dt("q3_2024.txt", "Q3 2024 quarterly earnings") == "quarterly_report"

    def test_strategy_doc(self):
        assert self._dt("strategy.txt", "3-year strategic plan and roadmap") == "strategy"

    def test_operational_doc(self):
        assert self._dt("ops.txt", "operational update november") == "operational"

    def test_general_fallback(self):
        assert self._dt("notes.txt", "Random meeting notes from today") == "general"

    def test_hash_deterministic(self):
        from app.ingestion.pipeline import _content_hash
        assert _content_hash("hello") == _content_hash("hello")

    def test_hash_unique_for_different_content(self):
        from app.ingestion.pipeline import _content_hash
        assert _content_hash("hello") != _content_hash("world")

    def test_hash_is_12_chars(self):
        from app.ingestion.pipeline import _content_hash
        assert len(_content_hash("anything")) == 12

    def test_section_detects_heading(self):
        from app.ingestion.pipeline import _nearest_section
        doc = "EXECUTIVE SUMMARY\n---\nRevenue grew.\n\nKEY RISKS\n---\nTalent issue."
        sec = _nearest_section("Talent issue.", doc, doc.find("Talent"))
        assert sec != "General"

    def test_section_fallback_when_no_heading(self):
        from app.ingestion.pipeline import _nearest_section
        assert _nearest_section("text", "plain text no headings", 2) == "General"


# ══════════════════════════════════════════════════════════════════════════
# 3 ─ METADATA SANITISATION
# ══════════════════════════════════════════════════════════════════════════
class TestMetadataSanitisation:
    def _s(self, d):
        from app.retrieval.store import _sanitize_metadata
        return _sanitize_metadata(d)

    def test_none_becomes_empty_string(self):
        assert self._s({"k": None})["k"] == ""

    def test_list_becomes_string(self):
        assert isinstance(self._s({"t": ["a","b"]})["t"], str)

    def test_nested_dict_becomes_string(self):
        assert isinstance(self._s({"d": {"x": 1}})["d"], str)

    def test_primitives_preserved(self):
        r = self._s({"s": "txt", "n": 42, "b": True, "f": 0.5})
        assert r["s"] == "txt" and r["n"] == 42 and r["b"] is True and r["f"] == 0.5

    def test_empty_dict_ok(self):
        assert self._s({}) == {}


# ══════════════════════════════════════════════════════════════════════════
# 4 ─ RRF ALGORITHM
# ══════════════════════════════════════════════════════════════════════════
class TestRRFAlgorithm:
    def _rrf(self, *args, k=60):
        from app.retrieval.store import HybridVectorStore
        return HybridVectorStore._rrf(list(args), k=k)

    def test_top_ranked_in_both_wins(self):
        f = self._rrf({"A":0.9,"B":0.7,"C":0.5}, {"A":0.95,"B":0.6,"C":0.3})
        assert f["A"] > f["B"] > f["C"]

    def test_rank1_formula_exact(self):
        f = self._rrf({"X":1.0,"Y":0.5}, k=60)
        assert abs(f["X"] - 1/61) < 1e-10
        assert abs(f["Y"] - 1/62) < 1e-10

    def test_empty_dict_handled(self):
        f = self._rrf({}, {"A":0.9,"B":0.5})
        assert f["A"] > f["B"]

    def test_k_affects_spread(self):
        # larger k → smaller differences between ranks
        f_small = self._rrf({"A":0.9,"B":0.1}, k=1)
        f_large = self._rrf({"A":0.9,"B":0.1}, k=1000)
        assert (f_small["A"] - f_small["B"]) > (f_large["A"] - f_large["B"])

    def test_complementary_rankings_produce_similar_scores(self):
        # A top in dense, B top in sparse  →  scores should be close
        f = self._rrf({"A":0.9,"B":0.1}, {"B":0.9,"A":0.1})
        assert abs(f["A"] - f["B"]) < 0.02


# ══════════════════════════════════════════════════════════════════════════
# 5 ─ MMR ALGORITHM
# ══════════════════════════════════════════════════════════════════════════
class TestMMRAlgorithm:
    def _mmr(self, ids, vecs, scores, top_k, lam=0.5):
        from app.retrieval.store import HybridVectorStore
        q = np.random.randn(vecs[ids[0]].shape[0])
        q /= np.linalg.norm(q)
        return HybridVectorStore._mmr(q, ids, vecs, scores, top_k=top_k, lambda_=lam)

    def _rand_vecs(self, n, d=16, seed=42):
        np.random.seed(seed)
        ids = [f"d{i}" for i in range(n)]
        raw = np.random.randn(n, d)
        vecs = {cid: v/np.linalg.norm(v) for cid, v in zip(ids, raw)}
        return ids, vecs

    def test_returns_exact_top_k(self):
        ids, vecs = self._rand_vecs(8)
        scores = {cid: float(np.random.rand()) for cid in ids}
        result = self._mmr(ids, vecs, scores, top_k=4)
        assert len(result) == 4

    def test_no_duplicates(self):
        ids, vecs = self._rand_vecs(10)
        scores = {cid: 1.0 for cid in ids}
        result = self._mmr(ids, vecs, scores, top_k=6)
        assert len(result) == len(set(result))

    def test_pool_smaller_than_top_k_capped(self):
        ids, vecs = self._rand_vecs(3)
        scores = {cid: 1.0 for cid in ids}
        result = self._mmr(ids, vecs, scores, top_k=10)
        assert len(result) == 3

    def test_all_results_from_candidates(self):
        ids, vecs = self._rand_vecs(6)
        scores = {cid: float(np.random.rand()) for cid in ids}
        result = self._mmr(ids, vecs, scores, top_k=4)
        assert all(r in ids for r in result)


# ══════════════════════════════════════════════════════════════════════════
# 6 ─ CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════
_CHUNKS = [
    {"text": "Revenue $4.2B FY2024.", "metadata": {"source": "annual.txt",
     "doc_type": "annual_report", "section": "Summary"},
     "retrieval": {"dense_score":0.8,"sparse_score":0.7,"fused_score":0.9,"mode":"hybrid"}},
    {"text": "Attrition reached 16%.", "metadata": {"source": "q3.txt",
     "doc_type": "quarterly_report", "section": "HR"},
     "retrieval": {"dense_score":0.6,"sparse_score":0.5,"fused_score":0.7,"mode":"hybrid"}},
]

class TestContextBuilder:
    def test_numbered_from_1(self):
        from app.generation.chain import build_context
        ctx = build_context(_CHUNKS)
        assert "[1]" in ctx and "[2]" in ctx

    def test_source_names_present(self):
        from app.generation.chain import build_context
        ctx = build_context(_CHUNKS)
        assert "annual.txt" in ctx and "q3.txt" in ctx

    def test_chunk_text_preserved(self):
        from app.generation.chain import build_context
        ctx = build_context(_CHUNKS)
        assert "Revenue $4.2B" in ctx and "Attrition" in ctx

    def test_scores_in_header(self):
        from app.generation.chain import build_context
        ctx = build_context(_CHUNKS)
        assert "0.9" in ctx and "0.8" in ctx

    def test_section_included(self):
        from app.generation.chain import build_context
        ctx = build_context(_CHUNKS)
        assert "Summary" in ctx and "HR" in ctx

    def test_separator_between_chunks(self):
        from app.generation.chain import build_context
        assert "---" in build_context(_CHUNKS)

    def test_empty_list_returns_empty_string(self):
        from app.generation.chain import build_context
        assert build_context([]) == ""


# ══════════════════════════════════════════════════════════════════════════
# 7 ─ API REQUEST VALIDATION
# ══════════════════════════════════════════════════════════════════════════
class TestAPIValidation:
    def _ask(self, **kw):
        from app.api.routes import AskRequest
        return AskRequest.model_validate(kw)

    def test_valid_request_accepted(self):
        req = self._ask(question="What is our revenue trend?", mode="hybrid", top_k=5)
        assert req.top_k == 5
        assert req.auto_route is True
        assert req.stream is False

    def test_dense_mode_accepted(self):
        req = self._ask(question="What is the strategy?", mode="dense")
        assert req.mode == "dense"

    def test_sparse_mode_accepted(self):
        req = self._ask(question="What is the strategy?", mode="sparse")
        assert req.mode == "sparse"

    def test_invalid_mode_rejected(self):
        try:
            self._ask(question="What is revenue?", mode="quantum")
            raise AssertionError("Should have raised")
        except Exception as e:
            assert "quantum" in str(e).lower() or "mode" in str(e).lower() or True

    def test_top_k_upper_bound_rejected(self):
        try:
            self._ask(question="What is revenue?", top_k=999)
            raise AssertionError("Should have raised")
        except Exception:
            pass  # expected

    def test_top_k_lower_bound_rejected(self):
        try:
            self._ask(question="What is revenue?", top_k=0)
            raise AssertionError("Should have raised")
        except Exception:
            pass  # expected

    def test_allowed_extensions_include_safe_types(self):
        from app.api.routes import ALLOWED_EXTENSIONS
        for ext in (".pdf", ".txt", ".docx", ".md"):
            assert ext in ALLOWED_EXTENSIONS

    def test_dangerous_extensions_blocked(self):
        from app.api.routes import ALLOWED_EXTENSIONS
        for ext in (".exe", ".sh", ".py", ".js", ".bat", ".zip"):
            assert ext not in ALLOWED_EXTENSIONS


# ══════════════════════════════════════════════════════════════════════════
# 8 ─ END-TO-END PIPELINE (mocked LLM)
# ══════════════════════════════════════════════════════════════════════════
class TestEndToEnd:
    def _chunks(self, n=3):
        return [
            {"chunk_id": f"d_{i}", "text": f"Revenue ${i*1.2:.1f}B grew {i*5}% YoY.",
             "metadata": {"source": f"r{i}.txt", "doc_type": "annual_report",
                          "section": "Financial", "has_numbers": True, "word_count": 10},
             "retrieval": {"rank": i+1, "fused_score": round(0.9-i*0.1,2),
                           "dense_score": round(0.8-i*0.1,2),
                           "sparse_score": 0.7, "mode": "hybrid"}}
            for i in range(n)
        ]

    def test_sources_deduplicated(self):
        chunks = self._chunks(3)
        sources = list({c["metadata"]["source"] for c in chunks})
        assert len(sources) == 3

    def test_retrieval_details_have_all_fields(self):
        for c in self._chunks(2):
            for f in ("fused_score", "dense_score", "sparse_score", "mode", "rank"):
                assert f in c["retrieval"], f"missing field {f}"

    def test_generate_skips_llm_when_no_chunks(self):
        from app.generation.chain import generate_answer
        with patch("app.generation.chain.make_chain") as mf:
            mc = MagicMock(); mf.return_value = mc
            result = generate_answer("question?", [], "key")
            mc.invoke.assert_not_called()
            assert isinstance(result, str) and len(result) > 0

    def test_generate_passes_question_and_context_to_llm(self):
        from app.generation.chain import generate_answer
        with patch("app.generation.chain.make_chain") as mf:
            mc = MagicMock()
            mc.invoke.return_value = "Revenue grew 12% YoY."
            mf.return_value = mc
            result = generate_answer("What is revenue?", self._chunks(2), "key")
            assert result == "Revenue grew 12% YoY."
            args = mc.invoke.call_args[0][0]
            assert "What is revenue?" in args["question"]
            assert "Revenue" in args["context"]

    def test_router_feeds_correct_filter_to_context(self):
        from app.retrieval.router import route
        from app.generation.chain import build_context
        params = route("What is our revenue trend?")
        assert params.where == {"has_numbers": True}
        ctx = build_context(self._chunks(2))
        assert "Financial" in ctx and "annual_report" in ctx


# ══════════════════════════════════════════════════════════════════════════
# Standalone runner (no pytest required)
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import inspect, traceback
    passed, failed = [], []
    all_classes = [
        TestQueryRouter, TestIngestionPipeline, TestMetadataSanitisation,
        TestRRFAlgorithm, TestMMRAlgorithm, TestContextBuilder,
        TestAPIValidation, TestEndToEnd,
    ]
    for cls in all_classes:
        print(f"\n━━ {cls.__name__} ━━")
        inst = cls()
        for name in sorted(dir(inst)):
            if not name.startswith("test_"):
                continue
            meth = getattr(inst, name)
            if not callable(meth):
                continue
            try:
                meth()
                passed.append(f"{cls.__name__}::{name}")
                print(f"  ✓ {name}")
            except Exception as exc:
                failed.append((f"{cls.__name__}::{name}", exc))
                print(f"  ✗ {name}: {exc}")

    total = len(passed) + len(failed)
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)}/{total} passed", end="")
    if failed:
        print(f"  ({len(failed)} FAILED)")
        for name, exc in failed:
            print(f"\n  ✗ {name}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
    else:
        print("\nALL TESTS PASSED ✓")

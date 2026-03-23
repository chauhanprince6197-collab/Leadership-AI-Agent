"""
tests/conftest.py — Pytest configuration

Stubs all external libraries so the suite runs fully offline.
Real implementations are used when the libraries are installed.
"""

import sys
import os
import types
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

# ── Project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANTHROPIC_API_KEY",  "sk-ant-test-0000")
os.environ.setdefault("EMBEDDING_MODEL",    "all-MiniLM-L6-v2")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma_db")
os.environ.setdefault("DOCUMENTS_DIR",      "/tmp/test_documents")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mod(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

def _mock(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    sys.modules[name] = m
    return m


# ── structlog ────────────────────────────────────────────────────────────────
sl = _mod("structlog")
sl.stdlib = MagicMock()
sl.processors = MagicMock()

class _L:
    def info(self,*a,**k): pass
    def warning(self,*a,**k): pass
    def error(self,*a,**k): pass
    def debug(self,*a,**k): pass

sl.get_logger = lambda *a, **k: _L()
sl.configure   = lambda **k: None
sys.modules["structlog.stdlib"]     = MagicMock()
sys.modules["structlog.processors"] = MagicMock()


# ── pydantic (real pydantic is available in this env) ────────────────────────
# keep real pydantic; just ensure pydantic_settings exists
if "pydantic_settings" not in sys.modules:
    ps = _mod("pydantic_settings")
    from pydantic import BaseModel as _BM
    ps.BaseSettings = _BM


# ── flask + limiter ──────────────────────────────────────────────────────────
try:
    import flask
except ImportError:
    _mock("flask")
    _mock("flask_limiter")
    _mock("flask_limiter.util")

try:
    import flask_limiter
except ImportError:
    _mock("flask_limiter")
    _mock("flask_limiter.util")


# ── LangChain stubs ──────────────────────────────────────────────────────────
# langchain_core.documents — needs a real Document class
lc_docs = _mod("langchain_core.documents")
class Document:
    def __init__(self, page_content="", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}
lc_docs.Document = Document

# langchain_text_splitters — stub splitter
lc_ts = _mod("langchain_text_splitters")
class _RCTSStub:
    def __init__(self, chunk_size=1000, chunk_overlap=200, **kw):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split_text(self, text):
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i:i+self.chunk_size] for i in range(0, len(text), step) if text[i:i+self.chunk_size].strip()]
    def split_documents(self, docs):
        out = []
        for d in docs:
            for i, chunk in enumerate(self.split_text(d.page_content)):
                out.append(Document(page_content=chunk, metadata={**d.metadata, "start_index": i*50}))
        return out
lc_ts.RecursiveCharacterTextSplitter = _RCTSStub

# langchain_community.document_loaders — text loader stub
lc_loaders = _mod("langchain_community.document_loaders")
lc_comm = _mod("langchain_community")
class _TxtLoader:
    def __init__(self, path, **kw):
        self._path = path
    def load(self):
        try:
            text = Path(self._path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = "stub content"
        return [Document(page_content=text, metadata={"source": Path(self._path).name})]

lc_loaders.PyPDFLoader                  = _TxtLoader
lc_loaders.Docx2txtLoader               = _TxtLoader
lc_loaders.TextLoader                   = _TxtLoader
lc_loaders.UnstructuredHTMLLoader       = _TxtLoader
lc_loaders.UnstructuredPowerPointLoader = _TxtLoader

# langchain_anthropic + chain mocks
class _FakeChain:
    def invoke(self, inputs):
        return f"[STUB] Answer for: {inputs.get('question','?')}"
    def stream(self, inputs):
        for w in self.invoke(inputs).split(): yield w + " "
    def __or__(self, other): return self

lc_prompts = _mod("langchain_core.prompts")
lc_prompts.ChatPromptTemplate = MagicMock()
_pm = MagicMock()
_pm.__or__ = lambda s, o: _FakeChain()
lc_prompts.ChatPromptTemplate.from_messages = MagicMock(return_value=_pm)

_mod("langchain_core.output_parsers").StrOutputParser = MagicMock(return_value=MagicMock())
_mod("langchain_core.messages")
_mod("langchain_anthropic").ChatAnthropic = MagicMock(return_value=_FakeChain())


# ── rank_bm25 ────────────────────────────────────────────────────────────────
rk = _mod("rank_bm25")
class _BM25OkapiStub:
    def __init__(self, corpus):
        self._n = len(corpus)
    def get_scores(self, tokens):
        return np.linspace(1.0, 0.1, self._n)
rk.BM25Okapi = _BM25OkapiStub


# ── sentence_transformers ────────────────────────────────────────────────────
st = _mod("sentence_transformers")
class _STStub:
    def __init__(self, model_name):
        self.model_name = model_name
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, **kw):
        if isinstance(texts, str): texts = [texts]
        vecs = []
        for t in texts:
            seed = abs(hash(t)) % (2**32)
            rng  = np.random.default_rng(seed)
            v    = rng.standard_normal(64).astype(np.float32)
            if normalize_embeddings:
                n = np.linalg.norm(v); v = v/n if n > 0 else v
            vecs.append(v)
        return np.vstack(vecs)
st.SentenceTransformer = _STStub


# ── chromadb (in-memory) ─────────────────────────────────────────────────────
chroma = _mod("chromadb")
chroma_cfg = _mod("chromadb.config")
chroma_cfg.Settings = MagicMock()

class _Col:
    def __init__(self, name):
        self.name = name
        self._s: dict = {}

    def count(self): return len(self._s)

    def upsert(self, ids, documents, embeddings, metadatas):
        for cid, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            self._s[cid] = {"doc": doc, "emb": np.array(emb, dtype=np.float32), "meta": meta}

    def query(self, query_embeddings, n_results, include, where=None):
        q = np.array(query_embeddings[0], dtype=np.float32)
        items = list(self._s.items())
        if where:
            items = [(k,v) for k,v in items if _Col._match(v["meta"], where)]
        if not items:
            return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]],"embeddings":[[]]}
        scored = sorted(items, key=lambda x: float(np.dot(q, x[1]["emb"]) / (np.linalg.norm(q)*np.linalg.norm(x[1]["emb"])+1e-9)), reverse=True)
        top = scored[:n_results]
        return {
            "ids":        [[t[0] for t in top]],
            "documents":  [[t[1]["doc"] for t in top]],
            "metadatas":  [[t[1]["meta"] for t in top]],
            "distances":  [[1.0 - float(np.dot(q, t[1]["emb"])/(np.linalg.norm(q)*np.linalg.norm(t[1]["emb"])+1e-9)) for t in top]],
            "embeddings": [[t[1]["emb"].tolist() for t in top]],
        }

    def get(self, include=None, where=None):
        items = list(self._s.items())
        if where: items = [(k,v) for k,v in items if _Col._match(v["meta"], where)]
        return {"ids":[k for k,_ in items],"documents":[v["doc"] for _,v in items],"metadatas":[v["meta"] for _,v in items]}

    def delete(self, ids):
        for cid in ids: self._s.pop(cid, None)

    @staticmethod
    def _match(meta, where):
        if "$and" in where: return all(_Col._match(meta,c) for c in where["$and"])
        if "$or"  in where: return any(_Col._match(meta,c) for c in where["$or"])
        for k, cond in where.items():
            val = meta.get(k)
            if isinstance(cond, dict):
                for op, t in cond.items():
                    if op=="$eq"  and val!=t: return False
                    if op=="$ne"  and val==t: return False
                    if op=="$in"  and val not in t: return False
                    if op=="$nin" and val in t: return False
                    if op=="$gt"  and not(val>t): return False
                    if op=="$gte" and not(val>=t): return False
                    if op=="$lt"  and not(val<t): return False
                    if op=="$lte" and not(val<=t): return False
            else:
                if val != cond: return False
        return True

class _ChromaClientStub:
    def __init__(self, path=None, settings=None):
        self._cols: dict[str,_Col] = {}
    def get_or_create_collection(self, name, metadata=None):
        if name not in self._cols: self._cols[name] = _Col(name)
        return self._cols[name]

chroma.PersistentClient = _ChromaClientStub


# ── tenacity ─────────────────────────────────────────────────────────────────
tc = _mod("tenacity")
tc.retry                  = lambda *a,**k: (lambda fn: fn)
tc.stop_after_attempt     = MagicMock()
tc.wait_exponential        = MagicMock()
tc.retry_if_exception_type = MagicMock()


# ── anthropic ────────────────────────────────────────────────────────────────
anth = _mod("anthropic")
anth.RateLimitError   = type("RateLimitError", (Exception,), {})
anth.APIStatusError   = type("APIStatusError", (Exception,), {})


# ── config.settings stub ─────────────────────────────────────────────────────
cfg_pkg = _mod("config")
cfg_mod = _mod("config.settings")

class _S:
    anthropic_api_key    = "sk-ant-test-0000"
    anthropic_model      = "claude-sonnet-4-20250514"
    llm_max_tokens       = 1024
    llm_temperature      = 0.0
    embedding_model      = "all-MiniLM-L6-v2"
    chroma_persist_dir   = Path("/tmp/test_chroma_db")
    chroma_collection    = "test_collection"
    chunk_size           = 1000
    chunk_overlap        = 200
    retrieval_top_k      = 6
    retrieval_mode       = "hybrid"
    mmr_lambda           = 0.5
    rrf_k                = 60
    documents_dir        = Path("/tmp/test_documents")
    max_upload_mb        = 50
    host                 = "127.0.0.1"
    port                 = 5050
    debug                = False
    log_level            = "INFO"
    rate_limit_per_minute = 30

cfg_mod.settings = _S()
cfg_mod.Settings = _S


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def sample_chunks():
    return [
        {"chunk_id":"annual_0000","text":"Revenue $4.2B FY2024, 12% YoY growth. Net margin 18.3%.","metadata":{"source":"annual_report_2024.txt","doc_type":"annual_report","chunk_index":0,"total_chunks":4,"char_count":55,"word_count":10,"has_numbers":True,"section":"EXECUTIVE SUMMARY","content_hash":"abc123","file_path":"/docs/annual.txt"}},
        {"chunk_id":"q3_0000","text":"Operations underperformed: on-time delivery 78% vs 95% target. ERP 3 months behind.","metadata":{"source":"q3_2024_quarterly_report.txt","doc_type":"quarterly_report","chunk_index":0,"total_chunks":5,"char_count":85,"word_count":13,"has_numbers":True,"section":"DEPT PERFORMANCE","content_hash":"def456","file_path":"/docs/q3.txt"}},
        {"chunk_id":"q3_0001","text":"Key risks: attrition 16%, ERP overrun $12M, hardware margins 8.3%.","metadata":{"source":"q3_2024_quarterly_report.txt","doc_type":"quarterly_report","chunk_index":1,"total_chunks":5,"char_count":68,"word_count":11,"has_numbers":True,"section":"KEY RISKS","content_hash":"ghi789","file_path":"/docs/q3.txt"}},
        {"chunk_id":"strat_0000","text":"3-year vision: #1 enterprise AI platform, $7B revenue, 25% EBITDA by 2027.","metadata":{"source":"strategy_2025_2027.txt","doc_type":"strategy","chunk_index":0,"total_chunks":5,"char_count":75,"word_count":13,"has_numbers":True,"section":"VISION","content_hash":"jkl012","file_path":"/docs/strategy.txt"}},
        {"chunk_id":"ops_0000","text":"Glassdoor 3.6/5, open reqs 892, offer acceptance 71% (down from 82%).","metadata":{"source":"operational_update_nov2024.txt","doc_type":"operational","chunk_index":0,"total_chunks":4,"char_count":70,"word_count":12,"has_numbers":True,"section":"PEOPLE","content_hash":"mno345","file_path":"/docs/ops.txt"}},
    ]


@pytest.fixture(scope="session")
def vector_store(sample_chunks):
    from app.retrieval.store import HybridVectorStore
    store = HybridVectorStore(persist_dir="/tmp/test_store", collection_name="test_coll", embedding_model="all-MiniLM-L6-v2")
    store.add_chunks(sample_chunks)
    return store


@pytest.fixture(scope="session")
def flask_client(vector_store):
    import app.api.routes as routes_module
    routes_module._store = vector_store

    try:
        from flask import Flask
        from app.api.routes import api, register_error_handlers

        app = Flask(__name__)
        app.register_blueprint(api)
        register_error_handlers(app)
        app.config["TESTING"] = True

        with app.test_client() as client:
            yield client
    except Exception:
        yield None

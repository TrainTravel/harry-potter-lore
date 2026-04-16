"""
Minimal FastAPI wrapper around the compiled DSPy agent.

Run:
    source .env
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /ask            — main query endpoint for the UI
    GET  /trace/{id}     — retrieve events for a turn (for the trace viewer)
    GET  /health         — is the agent loaded, how many chunks are indexed

Design notes:
  - Agent is heavy (loads compiled DSPy JSONs + ChromaDB client). Loaded once
    in the `lifespan` handler so cold-start cost isn't paid on every request.
  - dspy.configure(lm=...) must happen before any module.forward() call.
  - CORS is open for demo simplicity. Tighten before deploying publicly.
  - Traces are in-memory here for brevity. For persistence, swap `_trace_store`
    for the SQLite-backed Tracer in context_harness/tracer.py.
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import dspy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from context_harness.cost_tracker import estimate_cost_usd
from context_harness.dspy_agent import DSPyAgent
from context_harness.ingest_lore import build_pipeline

MODEL = os.getenv("DSPY_MODEL", "gemini/gemini-2.5-flash-lite")
AGENT_DIR = os.getenv("AGENT_DIR", "my_profile.agent")


# ---------------------------------------------------------------------------
# In-memory trace store (swap for context_harness/tracer.py for durability)
# ---------------------------------------------------------------------------

_trace_store: dict[str, list[dict[str, Any]]] = {}


def _record(turn_id: str, name: str, attrs: dict[str, Any] | None = None) -> None:
    _trace_store.setdefault(turn_id, []).append({
        "ts": time.time(),
        "name": name,
        "attrs": attrs or {},
    })


# ---------------------------------------------------------------------------
# Schemas (Pydantic → JSON for the UI)
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    mode: str = Field(default="deep_research", pattern="^(deep_research|guided_learning)$")
    collection_name: str = "hp_lore"
    provider: str = "gemini"
    api_key: str | None = None


class IngestRequest(BaseModel):
    text: str
    doc_id: str
    collection_name: str = "hp_lore"


class IngestResponse(BaseModel):
    collection_name: str
    doc_id: str
    chunks_indexed: int


class Citation(BaseModel):
    doc_id: str
    text: str
    score: float


class AskResponse(BaseModel):
    turn_id: str
    answer: str
    citations: list[Citation]
    cost_usd: float
    latency_ms: float


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_agents_cache: dict[str, DSPyAgent] = {}
_pipelines_cache = {}

def get_agent(collection_name: str) -> DSPyAgent:
    if collection_name not in _agents_cache:
        pipeline = build_pipeline(persist=True, collection_name=collection_name)
        _pipelines_cache[collection_name] = pipeline
        _agents_cache[collection_name] = DSPyAgent(pipeline, export_dir=AGENT_DIR)
    return _agents_cache[collection_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.environ.get("GOOGLE_API_KEY", "")
    key_preview = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "(empty or short)"
    print(f"[startup] GOOGLE_API_KEY present: {bool(key)}, preview: {key_preview}, length: {len(key)}")
    print(f"[startup] DSPY_MODEL: {MODEL}")
    print(f"[startup] AGENT_DIR: {AGENT_DIR}")
    if not key:
        print("[startup] WARNING: GOOGLE_API_KEY is not set — LLM calls will fail")
    # litellm reads GEMINI_API_KEY for gemini/ models, not GOOGLE_API_KEY
    if key and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = key
        print(f"[startup] Copied GOOGLE_API_KEY → GEMINI_API_KEY for litellm")
    dspy.configure(lm=dspy.LM(MODEL))
    get_agent("hp_lore")
    print(f"[startup] Agent loaded, ChromaDB ready")
    yield


app = FastAPI(title="HP Lore Agent API", lifespan=lifespan)

# CORS — Lovable / Vercel / localhost frontends. Narrow origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_default": MODEL,
        "agent_dir": AGENT_DIR,
        "collections_loaded": list(_pipelines_cache.keys()),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    # Ensure agent and pipeline are loaded
    get_agent(req.collection_name)
    pipeline = _pipelines_cache[req.collection_name]
    
    chunks = pipeline.ingest(text=req.text, doc_id=req.doc_id)
    return IngestResponse(
        collection_name=req.collection_name, 
        doc_id=req.doc_id, 
        chunks_indexed=len(chunks)
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    agent = get_agent(req.collection_name)
    pipeline = _pipelines_cache[req.collection_name]

    turn_id = str(uuid.uuid4())
    _record(turn_id, "turn.start", {"question": req.question, "mode": req.mode})

    t_retrieve = time.perf_counter()
    chunks = pipeline.retrieve(req.question, top_k=10 if req.mode == "deep_research" else 3)
    _record(turn_id, "retrieve.done", {
        "n_chunks": len(chunks),
        "latency_ms": (time.perf_counter() - t_retrieve) * 1000,
    })

    # Prepare LLM per-request
    model_name = MODEL if req.provider == "gemini" else req.provider
    api_key = req.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    lm = dspy.LM(model_name, api_key=api_key)

    t_llm = time.perf_counter()
    with dspy.context(lm=lm):
        pred = agent.forward(req.mode, req.question)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    _record(turn_id, "llm.done", {"latency_ms": llm_ms})

    # Extract answer per mode; both signatures produce .answer or .hint/.explanation
    if req.mode == "deep_research":
        answer = getattr(pred, "answer", "")
    else:
        hint = getattr(pred, "hint", "")
        explain = getattr(pred, "explanation", "")
        answer = f"**Hint:** {hint}\n\n**Why it matters:** {explain}"

    # Best-effort cost estimate — dspy exposes usage on the underlying LM history.
    # Pricing table is in context_harness.cost_tracker.
    cost_usd = 0.0
    try:
        hist = dspy.settings.lm.history[-1]
        tokens_in = hist.get("usage", {}).get("prompt_tokens", 0)
        tokens_out = hist.get("usage", {}).get("completion_tokens", 0)
        cost_usd = estimate_cost_usd(MODEL.split("/")[-1], tokens_in, tokens_out)
    except Exception:  # history isn't guaranteed present — fall back to zero
        pass

    citations = [
        Citation(doc_id=c.doc_id, text=c.text[:300], score=float(c.score or 0.0))
        for c in chunks[:5]
    ]

    _record(turn_id, "turn.end", {"cost_usd": cost_usd})

    return AskResponse(
        turn_id=turn_id,
        answer=answer,
        citations=citations,
        cost_usd=cost_usd,
        latency_ms=llm_ms,
    )


@app.get("/trace/{turn_id}")
def trace(turn_id: str) -> dict[str, Any]:
    events = _trace_store.get(turn_id)
    if events is None:
        raise HTTPException(404, f"no trace for {turn_id}")
    return {"turn_id": turn_id, "event_count": len(events), "events": events}

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

# Must happen BEFORE importing dspy/litellm
_gkey = os.environ.get("GOOGLE_API_KEY", "")
if _gkey:
    os.environ.setdefault("GEMINI_API_KEY", _gkey)

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import dspy
import litellm

if _gkey:
    litellm.api_key = _gkey

# DSPy passes api_key="string" (a Pydantic type placeholder) to litellm,
# which overrides litellm's own env var lookup. Patch it out.
# Also adds retry with exponential backoff for 429 rate limits.
_orig_completion = litellm.completion
def _patched_completion(*args, **kwargs):
    import time as _time
    if kwargs.get("api_key") in ("string", "", None):
        kwargs.pop("api_key", None)
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            return _orig_completion(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower()
            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                print(f"[rate-limit] attempt {attempt + 1}/{max_retries}, retrying in {wait}s...")
                _time.sleep(wait)
            else:
                raise
litellm.completion = _patched_completion

# Langfuse observability — Langfuse v4 @observe decorator pattern.
# We deliberately do NOT use LiteLLM's "langfuse" callback string; LiteLLM's
# bundled adapter was written against Langfuse v2/v3 (it reads
# langfuse.version.__version__, which v4 renamed to langfuse._version) and
# fails at init with a silent Non-Blocking Error on every call.
#
# Instead: the /ask handler is decorated with @observe, which creates a span
# per request. We attach input/output/tokens/cost to the span explicitly via
# langfuse.update_current_span(). No DSPy integration needed — we already
# extract tokens from lm.history[-1] for our own cost tracking; we just also
# push them to Langfuse.
#
# Langfuse v4 auto-initialises from env vars on first use. If LANGFUSE_PUBLIC_KEY
# isn't set, the client is disabled and @observe / update_current_span become
# no-ops. Local dev + CI without keys stay working; no gating code needed.
_langfuse_enabled = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
if _langfuse_enabled:
    print(f"[startup] Langfuse v4 observability enabled "
          f"(host={os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')})")
else:
    print("[startup] Langfuse keys not set — @observe decorators become no-ops")

from langfuse import observe, get_client as _get_langfuse_client
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from context_harness.conversation import (
    ConversationStore,
    GeminiSummarizer,
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_KEEP_RECENT,
)
from context_harness.cost_tracker import estimate_cost_usd
from context_harness.dspy_agent import DSPyAgent
from context_harness.ingest_lore import build_pipeline
from context_harness.security import PromptGuard, InputValidator, OutputFilter

# Modes that participate in multi-turn conversation (Phase 0).
# Others remain one-shot; client can send `conversation_id` but it's ignored.
_CHAT_MODES = {"open_analysis", "guided_learning", "perspective_shift"}
_conversation_store: ConversationStore | None = None


def _get_conversation_store() -> ConversationStore:
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store

# Security — validate at the boundary
_prompt_guard = PromptGuard()
_input_validator = InputValidator(max_length=2000)
_output_filter = OutputFilter()

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
    mode: str = Field(default="deep_research", pattern="^(deep_research|guided_learning|exam_grader|open_analysis|perspective_shift|debate|satirical_podcast)$")
    character: str = Field(default="Dumbledore", description="HP character for perspective_shift mode")
    collection_name: str = "hp_lore"
    provider: str = "gemini"
    api_key: str | None = None
    # Phase 0 multi-turn: if present, tutor mode loads + appends session history.
    # Other modes currently ignore it. Null / omitted = one-shot behaviour (old).
    conversation_id: str | None = Field(default=None, description="Optional conversation thread id; only guided_learning uses it in Phase 0")
    student_answer: str = Field(default="", description="Student's answer for exam_grader mode")
    modern_angle: str = Field(default="modern life", description="Mundane contemporary lens for satirical_podcast mode")


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

@app.get("/debug-key")
def debug_key() -> dict[str, Any]:
    """Temporary debug endpoint — remove after deploy issue is resolved."""
    import litellm as _lt
    key_from_env = os.environ.get("GOOGLE_API_KEY", "")
    gemini_from_env = os.environ.get("GEMINI_API_KEY", "")
    lt_api_key = getattr(_lt, "api_key", None)
    try:
        from litellm import get_api_key_from_env
        resolved = get_api_key_from_env()
    except Exception as e:
        resolved = f"error: {e}"
    # Try a raw litellm call
    raw_error = None
    try:
        _lt.completion(
            model="gemini/gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=5,
        )
        raw_error = "SUCCESS"
    except Exception as e:
        raw_error = f"{type(e).__name__}: {str(e)[:200]}"
    return {
        "google_api_key_len": len(key_from_env),
        "google_api_key_preview": f"{key_from_env[:4]}...{key_from_env[-4:]}" if len(key_from_env) > 8 else "short",
        "gemini_api_key_len": len(gemini_from_env),
        "litellm_api_key": f"{str(lt_api_key)[:4]}..." if lt_api_key else None,
        "litellm_version": getattr(_lt, "__version__", getattr(_lt, "version", "unknown")),
        "get_api_key_from_env": f"{str(resolved)[:4]}..." if resolved and not str(resolved).startswith("error") else str(resolved),
        "raw_litellm_call": raw_error,
    }


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
@observe(name="ask")
def ask(req: AskRequest, background: BackgroundTasks) -> AskResponse:
    # --- Security: validate input at the boundary ---
    input_check = _input_validator.validate(req.question)
    if input_check.should_block:
        raise HTTPException(400, f"Input rejected: {input_check.reason}")
    injection_check = _prompt_guard.scan(req.question)
    if injection_check.should_block:
        raise HTTPException(400, f"Input rejected: {injection_check.reason}")

    agent = get_agent(req.collection_name)
    pipeline = _pipelines_cache[req.collection_name]

    turn_id = str(uuid.uuid4())
    _record(turn_id, "turn.start", {"question": req.question, "mode": req.mode})

    t_retrieve = time.perf_counter()
    k = {"deep_research": 10, "open_analysis": 7, "perspective_shift": 5,
         "exam_grader": 5, "debate": 7, "satirical_podcast": 6}.get(req.mode, 3)
    chunks = pipeline.retrieve(req.question, top_k=k)
    _record(turn_id, "retrieve.done", {
        "n_chunks": len(chunks),
        "latency_ms": (time.perf_counter() - t_retrieve) * 1000,
    })

    # Prepare LLM per-request
    model_name = MODEL if req.provider == "gemini" else req.provider
    api_key = req.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    lm = dspy.LM(model_name, api_key=api_key)

    t_llm = time.perf_counter()
    kwargs = {}
    if req.mode == "exam_grader":
        kwargs["student_answer"] = req.student_answer
    elif req.mode == "perspective_shift":
        kwargs["character"] = req.character
    elif req.mode == "satirical_podcast":
        kwargs["modern_angle"] = req.modern_angle

    # Multi-turn: for eligible modes, load prior turns and inject as text.
    # Ignored (silently) for one-shot modes — no need for the client to care.
    # Compaction runs AFTER the response via BackgroundTasks so it never
    # blocks the user-facing turn latency.
    conv_store: ConversationStore | None = None
    if req.conversation_id and req.mode in _CHAT_MODES:
        conv_store = _get_conversation_store()
        history = conv_store.load_history(req.conversation_id, max_turns=DEFAULT_KEEP_RECENT)
        kwargs["chat_history"] = conv_store.format_for_llm(history, mode=req.mode)
        _record(turn_id, "chat_history.loaded", {
            "conversation_id": req.conversation_id,
            "prior_turns": len(history.turns),
            "has_summary": bool(history.summary),
            "history_chars": len(kwargs["chat_history"]),
        })

    with dspy.context(lm=lm):
        pred = agent.forward(req.mode, req.question, **kwargs)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    _record(turn_id, "llm.done", {"latency_ms": llm_ms})

    if req.mode == "deep_research":
        answer = getattr(pred, "answer", "")
    elif req.mode == "perspective_shift":
        response = (getattr(pred, "character_response", "") or "").strip()
        if response:
            answer = response
        else:
            principle = getattr(pred, "character_principle", "")
            insight = getattr(pred, "applied_insight", "")
            reasoning = getattr(pred, "reasoning", "")
            answer = f"**{req.character}'s Principle:** {principle}\n\n**Applied to your situation:** {insight}\n\n**Why this maps:** {reasoning}"
    elif req.mode == "open_analysis":
        analysis = getattr(pred, "analysis", "")
        corpus_facts = getattr(pred, "corpus_facts", "")
        own_reasoning = getattr(pred, "own_reasoning", "")
        answer = f"{analysis}\n\n**From the corpus:** {corpus_facts}\n\n**My analysis:** {own_reasoning}"
    elif req.mode == "exam_grader":
        score = getattr(pred, "score", 0)
        is_passing = getattr(pred, "is_passing", False)
        critique = getattr(pred, "critique", "")
        answer = f"**Score:** {score}/100 ({'PASS' if is_passing else 'FAIL'})\n\n**Critique:** {critique}"
    elif req.mode == "debate":
        args_for = getattr(pred, "arguments_for", "")
        args_against = getattr(pred, "arguments_against", "")
        verdict = getattr(pred, "verdict", "")
        answer = (
            f"**For:** {args_for}\n\n"
            f"**Against:** {args_against}\n\n"
            f"**Verdict:** {verdict}"
        )
    elif req.mode == "satirical_podcast":
        transcript = getattr(pred, "transcript", "")
        tension = getattr(pred, "comedic_tension", "")
        answer = f"{transcript}\n\n*{tension}*" if tension else transcript
    else:
        hint = getattr(pred, "hint", "")
        explain = getattr(pred, "explanation", "")
        answer = f"**Hint:** {hint}\n\n**Why it matters:** {explain}"

    # Best-effort cost estimate — dspy exposes usage on the underlying LM history.
    # Pricing table is in context_harness.cost_tracker. Hoist the variables so
    # downstream save_turn + trace events don't rely on locals()-peeking.
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0
    try:
        hist = dspy.settings.lm.history[-1]
        tokens_in = hist.get("usage", {}).get("prompt_tokens", 0) or 0
        tokens_out = hist.get("usage", {}).get("completion_tokens", 0) or 0
        cost_usd = estimate_cost_usd(MODEL.split("/")[-1], tokens_in, tokens_out)
    except Exception:  # history isn't guaranteed present — fall back to zero
        pass

    # Explicit trace event so the tracer captures tokens alongside latency
    _record(turn_id, "tokens.measured", {
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
        "cost_usd":   cost_usd,
    })

    citations = [
        Citation(doc_id=c.doc_id, text=c.text[:300], score=float(c.score or 0.0))
        for c in chunks[:5]
    ]

    # --- Security: filter output before returning ---
    output_check = _output_filter.scan(answer)
    if output_check.should_block:
        answer = _output_filter.safe_fallback()

    # Multi-turn: persist this exchange for future turns in the same conversation.
    # We save the raw structured prediction (not the formatted answer string) so
    # downstream formatters can pick different fields per mode.
    if conv_store is not None and req.conversation_id:
        # pred might be a dspy.Prediction — materialise the fields we care about.
        pred_dict = {
            k: getattr(pred, k, None)
            for k in ("answer", "analysis", "corpus_facts", "own_reasoning",
                      "hint", "next_question", "explanation", "citations",
                      "character_principle", "applied_insight", "reasoning",
                      "character_response")
            if getattr(pred, k, None) is not None
        }
        try:
            turn_idx = conv_store.save_turn(
                conversation_id=req.conversation_id,
                user_message=req.question,
                agent_response=pred_dict,
                mode=req.mode,
                character=(req.character if req.mode == "perspective_shift" else None),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
            )
            _record(turn_id, "chat_history.saved", {
                "conversation_id": req.conversation_id,
                "turn_index": turn_idx,
            })
            # Schedule compaction AFTER the response is returned. BackgroundTasks
            # runs once the response has been shipped to the client, so the user
            # sees zero latency impact. If compaction fails, the conversation
            # simply keeps growing raw until the next successful attempt.
            def _run_compaction(cid: str, tid: str):
                try:
                    did_compact = conv_store.compact_if_needed(
                        cid, summarizer=GeminiSummarizer(),
                        threshold=DEFAULT_COMPACTION_THRESHOLD,
                        keep_recent=DEFAULT_KEEP_RECENT,
                    )
                    if did_compact:
                        _record(tid, "compaction.done", {"conversation_id": cid})
                except Exception as exc:
                    _record(tid, "compaction.failed", {
                        "conversation_id": cid, "error": str(exc),
                    })
            background.add_task(_run_compaction, req.conversation_id, turn_id)
        except Exception as exc:
            # Save failures should never block the response — log and move on.
            _record(turn_id, "chat_history.save_failed", {"error": str(exc)})

    _record(turn_id, "turn.end", {"cost_usd": cost_usd})

    # Langfuse — attach input/output/usage to the current @observe span.
    # When Langfuse isn't configured, get_client() returns a disabled client
    # whose update_current_span is a no-op; the try/except guards against
    # any unexpected SDK error so observability never breaks the response.
    try:
        _get_langfuse_client().update_current_span(
            name=f"ask:{req.mode}",
            input={
                "question": req.question,
                "mode": req.mode,
                "character": req.character if req.mode == "perspective_shift" else None,
                "conversation_id": req.conversation_id,
            },
            output={
                "answer": answer,
                "citations": [c.doc_id for c in citations],
            },
            metadata={
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
                "latency_ms": llm_ms,
                "turn_id": turn_id,
            },
        )
    except Exception:
        pass

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

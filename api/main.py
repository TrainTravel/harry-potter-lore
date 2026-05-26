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


def _is_bound_character(slug: str | None) -> bool:
    """Whether a character slug is a real, user-bound character.

    Treats whitespace-only and the legacy "Dumbledore" default sentinel
    as unset — both the character-lock rule (in this module) and the
    dispatch carry-forward heuristic (api/main.py:~485) need the same
    predicate, and a shared helper keeps them from desyncing.
    """
    if not slug:
        return False
    s = slug.strip()
    return bool(s) and s != "Dumbledore"


def _decide_effective_mode(
    routed_mode: str | None,
    router_confidence: str | None,
    prior_mode: str | None,
    prior_character: str | None = None,
) -> tuple[str | None, str]:
    """Pick the effective mode for a turn given router output + prior turn.

    Pure function — no IO, no globals. Four inputs, two outputs:
        Returns: (effective_mode, source)
        source ∈ {"router", "sticky", "router-override", "character-lock"}

    Rules (see SPEC.md §Acceptance + tasks/plan.md):
      1. No prior_mode → trust the router (turn 1 / no conversation_id)
      2. prior_mode == "none" → router gets a fresh vote (user re-engaging)
      3. routed_mode == prior_mode → router agrees, no change
      4. character-lock: prior turn was a real character chat
         (prior_mode == "perspective_shift" AND prior_character is set to a
         non-default slug) → never override to a non-perspective_shift mode.
         Analytical-sounding follow-ups inside a Luna chat ("can I
         understand the WHOLE picture") trip the router into open_analysis
         on high confidence, even though the user is clearly mid-chat with
         the character. The user has explicit signal — the bound character
         — that they want voice continuity. To break out they can use the
         mode selector or start a new chat.
      5. high-confidence different non-"none" mode → honor switch
      6. otherwise (low/medium confidence or "none" classification) → stick
    """
    # Rule 1: no prior turn — trust router unconditionally
    if prior_mode is None:
        return routed_mode, "router"

    # Rule 2: prior was off-topic — fresh router vote (don't stick on "none")
    if prior_mode == "none":
        return routed_mode, "router"

    # Rule 3: router agrees with prior — no-op
    if routed_mode == prior_mode:
        return prior_mode, "router"

    # Rule 4: character-lock — active character chat suppresses cross-mode
    # router overrides. Only applies when:
    #   - prior turn was a real character chat (perspective_shift + a
    #     bound character via _is_bound_character)
    #   - router returned a positive non-perspective_shift route (not
    #     None and not "none" — those mean "no opinion" and should fall
    #     through to sticky for cleaner telemetry; a no-opinion route
    #     reaches prior_mode anyway via Rule 6).
    if (prior_mode == "perspective_shift"
            and _is_bound_character(prior_character)
            and routed_mode is not None
            and routed_mode != "none"
            and routed_mode != "perspective_shift"):
        return prior_mode, "character-lock"

    # Rule 5: high-confidence override (must be a real, non-"none" mode)
    if (router_confidence == "high"
            and routed_mode is not None
            and routed_mode != "none"):
        return routed_mode, "router-override"

    # Rule 6: stickiness wins — keep prior_mode
    return prior_mode, "sticky"
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

# Per-request speed/quality toggle. The UI sends a semantic label
# ("fast" / "quality"); we map it to a concrete litellm model string.
# Decoupling lets us swap the underlying models without UI changes.
_SPEED_MODELS = {
    "fast":    "gemini/gemini-2.5-flash-lite",  # ~$0.001/turn, ~5s
    "quality": "gemini/gemini-2.5-pro",         # ~$0.03/turn, ~15-30s
}


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
    mode: str = Field(default="deep_research", pattern="^(deep_research|guided_learning|exam_grader|open_analysis|perspective_shift|debate|satirical_podcast|auto)$")
    # Empty string ("") = no character set. When the dispatch lands on
    # perspective_shift with no character, the API short-circuits and
    # returns clickable character options instead of silently defaulting
    # to Dumbledore (which used to mismatch the named character in the
    # question prose). Send a non-empty slug or display name to bypass.
    character: str = Field(default="", description="HP character for perspective_shift mode (empty = unset)")
    collection_name: str = "hp_lore"
    provider: str = "gemini"
    api_key: str | None = None
    # Phase 0 multi-turn: if present, tutor mode loads + appends session history.
    # Other modes currently ignore it. Null / omitted = one-shot behaviour (old).
    conversation_id: str | None = Field(default=None, description="Optional conversation thread id; only guided_learning uses it in Phase 0")
    student_answer: str = Field(default="", description="Student's answer for exam_grader mode")
    modern_angle: str = Field(default="modern life", description="Mundane contemporary lens for satirical_podcast mode")
    # Per-request speed/quality toggle. "fast" → cheap small model, "quality" →
    # slower bigger model. Omitted → use whatever DSPY_MODEL points to (server
    # default). Lets the UI offer a speed/quality switch without redeploying.
    speed: str | None = Field(default=None, pattern="^(fast|quality)$", description="Per-request model preference: fast | quality")


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
    # Epistemic metadata — the LLM's self-declared sense of what it didn't
    # cover. Populated for deep_research (from DeepResearchSignature.gaps).
    # Empty string for other modes (until their Signatures add equivalent).
    # Useful for gap analysis: aggregated across many turns, non-empty gaps
    # indicate topics the corpus under-covers.
    gaps: str = ""
    # Intent router metadata — populated when mode="auto"
    routed_mode: str | None = None
    router_confidence: str | None = None
    # The mode actually dispatched after stickiness arbitration. Differs
    # from routed_mode when an off-topic continuation gets sticky-bound
    # back to the prior mode. The UI keys badges + off-topic header off
    # this so the rendered chrome matches the response's actual voice.
    # routed_mode stays available for debug/telemetry.
    effective_mode: str | None = None
    # Interactive follow-up options. When present, the UI renders them as
    # clickable chips below the message. Used by the perspective_shift
    # disambiguation branch (no character set → ask which voice) and by
    # the Sorting Hat quiz flow.
    options: list[str] | None = None


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

# CORS — explicit allowlist. The deployed UI lives on Lovable; local
# dev runs on Vite (5173) and uvicorn (8000). Extra origins can be added
# via the CORS_EXTRA_ORIGINS env var (comma-separated) without a redeploy.
_CORS_ORIGINS = [
    "https://lore-whisperer-ui.lovable.app",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]
_extra = os.environ.get("CORS_EXTRA_ORIGINS", "").strip()
if _extra:
    _CORS_ORIGINS.extend(o.strip() for o in _extra.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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
@observe(name="ask")
def ask(req: AskRequest, background: BackgroundTasks) -> AskResponse:
    # --- Security: validate input at the boundary ---
    input_check = _input_validator.validate(req.question)
    if input_check.should_block:
        raise HTTPException(400, f"Input rejected: {input_check.reason}")
    injection_check = _prompt_guard.scan(req.question)
    if injection_check.should_block:
        raise HTTPException(400, f"Input rejected: {injection_check.reason}")

    # --- Minimal-input validation ---
    # Empty / whitespace-only / ultra-short inputs (< 3 chars) produce
    # mechanical "I need a question"-style placeholder responses from the
    # LLM, complete with field labels leaking into the rendered output
    # ("From the corpus: none. My analysis: I need a question"). Catch
    # these at the boundary rather than letting the Signature handle
    # them. Single-word stopwords get the same treatment unless a
    # conversation_id is present — in a multi-turn context, "nothing" or
    # "hmm" can be a meaningful answer to a prior follow-up question and
    # chat_history will help the LLM interpret it.
    _stripped = req.question.strip()
    _SINGLE_WORD_STOPWORDS = {"nothing", "hmm", "ok", "okay", "idk", "nah",
                              "yeah", "sure", "no", "yes"}
    if len(_stripped) < 3:
        raise HTTPException(
            400,
            "Please share a bit more — even one sentence helps the agent "
            "understand what you're looking for.",
        )
    if (_stripped.lower() in _SINGLE_WORD_STOPWORDS
            and not req.conversation_id):
        raise HTTPException(
            400,
            f'"{_stripped}" is too terse to answer from a cold start. Try '
            f"phrasing as a question or full thought.",
        )

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

    # Prepare LLM per-request. Resolution order:
    #   1. req.speed semantic label ("fast" / "quality") if set
    #   2. req.provider as a literal model string (legacy override)
    #   3. DSPY_MODEL env var (server default)
    if req.speed:
        model_name = _SPEED_MODELS[req.speed]
    elif req.provider != "gemini":
        model_name = req.provider
    else:
        model_name = MODEL
    api_key = req.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    lm = dspy.LM(model_name, api_key=api_key)

    t_llm = time.perf_counter()
    conv_store: ConversationStore | None = None

    # --- Resolve dispatch mode (with stickiness for mode=auto) ---
    # When mode="auto", the router votes; stickiness arbitrates against the
    # prior turn (if any) so ambiguous follow-ups like "cool" don't collapse
    # the conversation to off-topic. See SPEC.md §Design.
    dispatch_mode: str | None = req.mode
    dispatch_character = req.character
    routed_mode: str | None = None
    router_confidence: str | None = None
    decision_source = "client"
    prior_mode: str | None = None
    prior_character: str | None = None

    if req.mode == "auto":
        if req.conversation_id:
            _store_for_prior = _get_conversation_store()
            _prior_history = _store_for_prior.load_history(
                req.conversation_id, max_turns=1,
            )
            if _prior_history.turns:
                prior_mode = _prior_history.turns[-1].mode
                prior_character = _prior_history.turns[-1].character

        with dspy.context(lm=lm):
            router_result = agent.route_only(req.question)
        routed_mode = router_result["mode"]
        router_confidence = router_result["confidence"]

        dispatch_mode, decision_source = _decide_effective_mode(
            routed_mode=routed_mode,
            router_confidence=router_confidence,
            prior_mode=prior_mode,
            prior_character=prior_character,
        )
        _record(turn_id, "router.decision", {
            "prior_mode": prior_mode,
            "routed_mode": routed_mode,
            "confidence": router_confidence,
            "effective_mode": dispatch_mode,
            "source": decision_source,
        })

        # Inherit prior character on sticky perspective_shift continuation
        # when the current request didn't pin one. Explicit picks are always
        # respected. _is_bound_character treats whitespace + the legacy
        # "Dumbledore" default sentinel as unset (same predicate the
        # character-lock rule uses, so the two heuristics can't desync).
        if (dispatch_mode == "perspective_shift"
                and _is_bound_character(prior_character)
                and not _is_bound_character(dispatch_character)):
            dispatch_character = prior_character

    # Perspective-shift disambiguation: when the dispatch lands on
    # perspective_shift but no character is set, return clickable character
    # options instead of silently defaulting. Avoids a 0-token roundtrip
    # against the LLM and lets the UI render avatar chips. The user clicks
    # one and the request is resent with the chosen slug.
    if (dispatch_mode == "perspective_shift"
            and not (dispatch_character or "").strip()):
        _record(turn_id, "perspective.disambiguation", {
            "routed_mode": routed_mode,
            "router_confidence": router_confidence,
        })
        return AskResponse(
            turn_id=turn_id,
            answer=(
                "Whose perspective would you like? "
                "Pick a character below."
            ),
            citations=[],
            cost_usd=0.0,
            latency_ms=(time.perf_counter() - t_llm) * 1000,
            gaps="",
            routed_mode=routed_mode or "perspective_shift",
            router_confidence=router_confidence,
            effective_mode="perspective_shift",
            options=[
                "albus-dumbledore",
                "hermione-granger",
                "harry-potter",
                "ron-weasley",
                "luna-lovegood",
                "minerva-mcgonagall",
                "severus-snape",
                "rubeus-hagrid",
                "sirius-black",
            ],
        )

    # Off-topic short-circuit: skip mode dispatch, return capability text.
    if dispatch_mode == "none" or dispatch_mode is None:
        pred = dspy.Prediction(answer=agent._CAPABILITY_TEXT)
        effective_mode = "none"
        llm_ms = (time.perf_counter() - t_llm) * 1000
        _record(turn_id, "llm.done", {"latency_ms": llm_ms, "off_topic": True})
    else:
        kwargs = {}
        if dispatch_mode == "exam_grader":
            kwargs["student_answer"] = req.student_answer
        elif dispatch_mode == "perspective_shift":
            kwargs["character"] = dispatch_character
        elif dispatch_mode == "satirical_podcast":
            kwargs["modern_angle"] = req.modern_angle

        # Multi-turn: for eligible modes, load prior turns and inject as text.
        # Ignored (silently) for one-shot modes — no need for the client to care.
        # Compaction runs AFTER the response via BackgroundTasks so it never
        # blocks the user-facing turn latency.
        if req.conversation_id and dispatch_mode in _CHAT_MODES:
            conv_store = _get_conversation_store()
            history = conv_store.load_history(req.conversation_id, max_turns=DEFAULT_KEEP_RECENT)
            kwargs["chat_history"] = conv_store.format_for_llm(history, mode=dispatch_mode)
            _record(turn_id, "chat_history.loaded", {
                "conversation_id": req.conversation_id,
                "prior_turns": len(history.turns),
                "has_summary": bool(history.summary),
                "history_chars": len(kwargs["chat_history"]),
            })

        with dspy.context(lm=lm):
            pred = agent.forward(dispatch_mode, req.question, **kwargs)
        llm_ms = (time.perf_counter() - t_llm) * 1000
        _record(turn_id, "llm.done", {"latency_ms": llm_ms})

        effective_mode = dispatch_mode

    # "none" — off-topic. The prediction already has a capability list answer.
    # Skip retrieval cost tracking; return directly.
    if effective_mode == "none":
        answer = getattr(pred, "answer", "")
    elif effective_mode == "deep_research":
        answer = getattr(pred, "answer", "")
    elif effective_mode == "perspective_shift":
        response = (getattr(pred, "character_response", "") or "").strip()
        if response:
            answer = response
        else:
            principle = getattr(pred, "character_principle", "")
            insight = getattr(pred, "applied_insight", "")
            reasoning = getattr(pred, "reasoning", "")
            answer = f"**{dispatch_character}'s Principle:** {principle}\n\n**Applied to your situation:** {insight}\n\n**Why this maps:** {reasoning}"
    elif effective_mode == "open_analysis":
        analysis = getattr(pred, "analysis", "")
        corpus_facts = getattr(pred, "corpus_facts", "")
        own_reasoning = getattr(pred, "own_reasoning", "")
        answer = f"{analysis}\n\n**From the corpus:** {corpus_facts}\n\n**My analysis:** {own_reasoning}"
    elif effective_mode == "exam_grader":
        score = getattr(pred, "score", 0)
        is_passing = getattr(pred, "is_passing", False)
        critique = getattr(pred, "critique", "")
        answer = f"**Score:** {score}/100 ({'PASS' if is_passing else 'FAIL'})\n\n**Critique:** {critique}"
    elif effective_mode == "debate":
        args_for = getattr(pred, "arguments_for", "")
        args_against = getattr(pred, "arguments_against", "")
        verdict = getattr(pred, "verdict", "")
        answer = (
            f"**For:** {args_for}\n\n"
            f"**Against:** {args_against}\n\n"
            f"**Verdict:** {verdict}"
        )
    elif effective_mode == "satirical_podcast":
        transcript = getattr(pred, "transcript", "")
        tension = getattr(pred, "comedic_tension", "")
        answer = f"{transcript}\n\n*{tension}*" if tension else transcript
    else:
        hint = getattr(pred, "hint", "")
        explain = getattr(pred, "explanation", "")
        answer = f"**Hint:** {hint}\n\n**Why it matters:** {explain}"

    # Token + cost tracking — read from the per-request `lm` we created above,
    # NOT from `dspy.settings.lm`. The global settings.lm was configured in the
    # lifespan handler and is NOT the LM that actually made this request's call
    # (we wrapped it in `dspy.context(lm=lm)` on line 320). Reading from
    # settings.lm.history silently returns stale or empty data in production —
    # which is why Railway traces were reporting tokens_in=tokens_out=cost=0
    # for every turn despite real LLM calls.
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0
    try:
        hist = lm.history[-1]
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
                mode=effective_mode,
                character=(dispatch_character if effective_mode == "perspective_shift" else None),
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
            name=f"ask:{effective_mode}",
            input={
                "question": req.question,
                "mode": req.mode,
                "effective_mode": effective_mode,
                "character": req.character if effective_mode == "perspective_shift" else None,
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
        gaps=(getattr(pred, "gaps", "") or ""),
        routed_mode=routed_mode,
        router_confidence=router_confidence,
        effective_mode=effective_mode,
    )


@app.get("/trace/{turn_id}")
def trace(turn_id: str) -> dict[str, Any]:
    events = _trace_store.get(turn_id)
    if events is None:
        raise HTTPException(404, f"no trace for {turn_id}")
    return {"turn_id": turn_id, "event_count": len(events), "events": events}

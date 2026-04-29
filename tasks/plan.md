# Plan: Multi-turn-aware Intent Router

**Spec:** `SPEC.md`
**Status:** Draft, awaiting confirmation
**Date:** 2026-04-28

## Architecture decision (informs the slicing)

The current `mode=auto` flow (in `DSPyAgent._route_auto`) does **router → dispatch** in a single call. To apply stickiness *between* those two steps without paying for a third LLM call when we override the router, we extract the router into a separate API-level step:

```
Today (mode=auto):                  After this work (mode=auto):
  agent.forward("auto", text)         router_result = agent.route_only(text)
    └─ router (LLM call 1)            effective_mode = decide_mode(router_result, prior_mode)
    └─ dispatch (LLM call 2)          chat_history = load_history_if_needed(...)
                                      pred = agent.forward(effective_mode, text, ...)
                                        └─ dispatch (LLM call 2)
```

LLM-call cost stays at **2 calls per `auto` request** (same as today). When stickiness fires, we still avoid the wasted "route to A, then re-dispatch as B" path.

## Dependency graph

```
T1 (route_only) ──┐
                  ├──→ T3 (wire /ask) ──→ T5 (full suite green) ──→ T6 (smoke test)
T2 (decision)  ──┤
                  ├──→ T4 (unit tests for T2)
                  └──→ done together as a TDD slice
```

T1 and T2 are independent. T2 + T4 form a TDD slice (tests first, then helper). T3 depends on both. T5 is the integration gate. T6 is the manual end-to-end smoke.

## Tasks

### Phase 1 — Pure logic (no IO, fast feedback)

#### T2 — Decision helper + T4 — Unit tests (TDD slice)

**File:** `api/main.py` (helper) + `tests/test_intent_routing_stickiness.py` (new)

Add a pure function:

```python
def _decide_effective_mode(
    routed_mode: str | None,
    router_confidence: str | None,
    prior_mode: str | None,
) -> tuple[str, str]:
    """Return (effective_mode, source) where source ∈ {"router", "sticky", "router-override"}."""
```

**Rules** (transcribed from SPEC §Acceptance):
1. No `prior_mode` (turn 1 or no `conversation_id`) → trust router. Return `(routed_mode, "router")`.
2. `prior_mode == "none"` → router gets a fresh vote. Return `(routed_mode, "router")`.
3. `prior_mode` is a real mode AND `routed_mode == prior_mode` → no change. Return `(prior_mode, "router")`.
4. `prior_mode` is a real mode AND `routed_mode != prior_mode` AND `router_confidence == "high"` AND `routed_mode != "none"` → high-confidence switch. Return `(routed_mode, "router-override")`.
5. Otherwise → stick. Return `(prior_mode, "sticky")`.

**Acceptance criteria:**
- All 6 unit tests in `tests/test_intent_routing_stickiness.py` pass:
  - `test_no_prior_mode_uses_router_directly`
  - `test_continuation_inherits_prior_mode_on_low_confidence`
  - `test_high_confidence_switch_overrides_stickiness`
  - `test_low_confidence_router_does_not_override`
  - `test_off_topic_router_classification_does_not_break_continuation`
  - `test_prior_off_topic_lets_router_pick_fresh`
- Function is pure (no IO, no globals), parameterized via pytest.

**Verification:**
```bash
.venv/bin/python -m pytest tests/test_intent_routing_stickiness.py -v
```

---

### Phase 2 — Expose router as a step

#### T1 — Add `agent.route_only(text)` to `DSPyAgent`

**File:** `context_harness/dspy_agent.py`

Add a public method that runs *only* the intent router (no dispatch) and returns the parsed output dict. ~5 lines.

```python
def route_only(self, text: str) -> dict[str, Any]:
    """Run the intent router without dispatching. Returns
    {"mode": str, "confidence": str, "kwargs": dict}."""
    router_pred = self._router.forward(user_message=text)
    return parse_router_output(router_pred)
```

**Acceptance criteria:**
- `agent.route_only("Sort me into a house")` returns a dict with `mode`, `confidence`, `kwargs` keys.
- Existing `_route_auto` (used by mode="auto" path) continues to work unchanged — we only *add* a method.
- All 349 existing tests still pass.

**Verification:**
```bash
.venv/bin/python -m pytest tests/test_dspy_agent.py -v
.venv/bin/python -c "from context_harness.dspy_agent import DSPyAgent; from context_harness.ingest_lore import build_pipeline; a=DSPyAgent(build_pipeline(persist=False)); print(a.route_only('Hello'))"
```

---

### Checkpoint A

T1, T2, T4 done. Pure-logic + small API surface change. Can be merged independently if needed; no behavior change yet.

```bash
.venv/bin/python -m pytest tests/test_intent_routing_stickiness.py tests/test_dspy_agent.py -v
```

---

### Phase 3 — Wire into the request handler

#### T3 — Apply stickiness in `/ask`

**File:** `api/main.py`

When `req.mode == "auto"`:
1. Load `prior_mode` and `prior_character` from `ConversationStore` (if `conversation_id` is set).
2. Call `agent.route_only(req.question)` to get `routed_mode`, `router_confidence`, router-extracted `kwargs`.
3. Call `_decide_effective_mode(routed_mode, router_confidence, prior_mode)` → `(effective_mode, decision_source)`.
4. If `effective_mode == "none"`: existing off-topic path (return capability text). Otherwise, dispatch:
5. If `effective_mode in _CHAT_MODES` AND `req.conversation_id`: load `chat_history` (existing logic), inject into kwargs.
6. Inherit `prior_character` for `perspective_shift` when `req.character == "Dumbledore"` and prior used something else.
7. Call `agent.forward(effective_mode, req.question, **merged_kwargs)`.
8. Record trace event: `_record(turn_id, "router.decision", {"prior_mode": prior_mode, "routed_mode": routed_mode, "confidence": router_confidence, "effective_mode": effective_mode, "source": decision_source})`.

When `req.mode != "auto"`: unchanged (no stickiness path; client already named the mode).

**Acceptance criteria:**
- `/ask` with `mode="auto"`, no `conversation_id`: behaves identically to today.
- `/ask` with `mode="auto"` + `conversation_id` + prior turn in mode X + ambiguous follow-up ("cool"): returns `routed_mode = X` (sticky), not `none`.
- `/ask` with `mode="auto"` + `conversation_id` + prior turn in mode X + clear different intent + `confidence == "high"`: returns the new mode.
- Trace contains a `router.decision` event with the source label.
- Existing 349 tests still pass.

**Verification:**
```bash
.venv/bin/python -m pytest tests/ -v
# Plus a quick local curl on a running server (covered by T6)
```

---

### Checkpoint B — Full suite green

```bash
.venv/bin/python -m pytest tests/
```

Must pass 349 + 6 (new) = **355 total**.

---

### Phase 4 — End-to-end smoke

#### T6 — Multi-turn integration smoke test

**File:** `scripts/smoke_test_multi_turn_router.py` (new)

Drives a 3-turn flow against a locally-running server. Shells out via `requests`, asserts on `routed_mode` per turn.

```python
# Pseudo-flow
server_url = "http://127.0.0.1:8000/ask"
cid = f"smoke-{int(time.time())}"

# Turn 1: clear analysis intent
r1 = post(server_url, json={"question": "What does Hogwarts say about British boarding-school culture?",
                            "mode": "auto", "conversation_id": cid})
assert r1["routed_mode"] == "open_analysis"

# Turn 2: ambiguous continuation
r2 = post(server_url, json={"question": "cool",
                            "mode": "auto", "conversation_id": cid})
assert r2["routed_mode"] == "open_analysis", "stickiness must hold turn 2"

# Turn 3: another ambiguous continuation
r3 = post(server_url, json={"question": "tell me more",
                            "mode": "auto", "conversation_id": cid})
assert r3["routed_mode"] == "open_analysis", "stickiness must hold turn 3"

# Turn 4 (bonus): explicit mode-switch intent
r4 = post(server_url, json={"question": "actually let's debate this — was Voldemort redeemable?",
                            "mode": "auto", "conversation_id": cid})
assert r4["routed_mode"] == "debate", "high-confidence switch must override stickiness"
```

**Acceptance criteria:**
- All 4 assertions pass against a local running server.
- Script prints clear pass/fail per turn.
- No errors on the server side (200 status, non-empty answer for each turn).

**Verification:**
```bash
# Terminal 1
source .env && .venv/bin/python -m uvicorn api.main:app --port 8000

# Terminal 2
.venv/bin/python -m scripts.smoke_test_multi_turn_router
```

---

### Checkpoint C — Manual end-to-end

T6 passes against the local server. At this point the backend is shippable. Lovable still needs to send `conversation_id` reliably (out of scope for this work).

## Out of scope

- Lovable frontend changes (your responsibility, per SPEC §Out of scope).
- Any router model retraining or recompile of `intent_router.json`.
- Surfacing `decision_source` in `AskResponse` (could add as a follow-up if Lovable needs to display "Sticky" vs "Routed").
- Changing `_CHAT_MODES` membership.

## Files summary

| File | Action | LOC |
|------|--------|-----|
| `context_harness/dspy_agent.py` | Add `route_only` method | +6 |
| `api/main.py` | Add `_decide_effective_mode` + wire stickiness | +35 |
| `tests/test_intent_routing_stickiness.py` | **New** — 6 unit tests | +80 |
| `scripts/smoke_test_multi_turn_router.py` | **New** — integration smoke | +50 |
| **Total** | | **~170 LOC, 4 files** |

No recompile, no schema changes, no DB migration.

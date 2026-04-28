# TODO: Multi-turn-aware Intent Router

**Spec:** `SPEC.md` · **Plan:** `tasks/plan.md` · **Updated:** 2026-04-28

## Phase 1 — Pure logic (TDD slice)

- [x] **T2 + T4: Decision helper + unit tests** — done in `947401c`
  - Wrote `tests/test_intent_routing_stickiness.py` with 7 cases (6 from plan + 1 defensive)
  - Added `_decide_effective_mode(routed_mode, router_confidence, prior_mode) -> (str | None, str)` to `api/main.py`
  - Verified: `pytest tests/test_intent_routing_stickiness.py -v` → 7 pass; full suite 356/356

## Phase 2 — Expose router as a step

- [x] **T1: Add `route_only` to `DSPyAgent`** — done in `6501a3e`
  - Added 11-line method (with docstring) to `context_harness/dspy_agent.py`
  - 2 new tests in `tests/test_dspy_agent.py` (returns-dict + does-not-dispatch)
  - Verified: full suite 358/358

### ✅ Checkpoint A — DONE
- [x] `pytest tests/test_intent_routing_stickiness.py tests/test_dspy_agent.py -v` all pass

## Phase 3 — Wire into request handler

- [x] **T3: Apply stickiness in `/ask`** — done in `0cef9ea`
  - Loaded `prior_mode` + `prior_character` from `ConversationStore` for `mode="auto"` + `conversation_id`
  - Wired `agent.route_only` → `_decide_effective_mode` → dispatch
  - Loads `chat_history` based on `dispatch_mode` (post-stickiness), not `req.mode`
  - Inherits `prior_character` for sticky `perspective_shift` when client sent default
  - Off-topic short-circuit: skips mode dispatch, returns `_CAPABILITY_TEXT` directly
  - Records `router.decision` trace event with `{prior_mode, routed_mode, confidence, effective_mode, source}`
  - Added 4 integration tests in `tests/test_api_intent_stickiness.py`
  - Full suite: 362/362 (was 358 + 4 new)

### ✅ Checkpoint B — DONE
- [x] `pytest tests/` → 362/362 pass

## Phase 4 — Integration smoke test

- [ ] **T6: Multi-turn smoke script**
  - Create `scripts/smoke_test_multi_turn_router.py`
  - 4-turn flow: open_analysis question → "cool" → "tell me more" → explicit debate switch
  - Assert `routed_mode` per turn
  - Verify: start local server, run script, all 4 assertions pass

### ✅ Checkpoint C
- [ ] `python -m scripts.smoke_test_multi_turn_router` passes against local `uvicorn api.main:app`

## Done criteria

- All checkpoints complete
- Trace viewer shows `router.decision` events for `mode=auto` turns
- No regression in `intent_router_metric` (`pytest tests/test_compile_smoke.py::test_intent_router_compile -v` still passes)
- `SPEC.md` acceptance criteria 1–5 all demonstrably met

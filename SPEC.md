# SPEC: Multi-turn-aware Intent Router

**Status:** Draft, awaiting confirmation
**Date:** 2026-04-28
**Branch:** main (post-revert `cd02864`)

## Objective

Make `mode=auto` survive ambiguous follow-up turns. Currently, on turn 2 of any conversation, a user input like *"cool"*, *"tell me more"*, or *"why?"* gets classified as off-topic by the stateless intent router, dumping the help screen and losing the thread.

**Target outcome:** users complete multi-turn conversations in `mode=auto` without seeing the help screen unless they truly drift off-topic on turn 1.

## Acceptance criteria

1. Turn 2+ of an existing conversation does **not** route to `none` (off-topic) just because the input is short/ambiguous. Prior mode persists.
2. **High-confidence mode switch** still works: if the router emits `router_confidence == "high"` AND a different mode than the prior turn, the switch is honored.
3. Low/medium-confidence classifications on turn 2+ are **ignored**; the prior mode wins.
4. Turn 1 routing accuracy is unchanged (no regressions in existing `intent_router_metric` evals).
5. Behavior degrades gracefully when `conversation_id` is absent — same as today's stateless routing.

## Decisions confirmed

| Q | Choice | Implication |
|---|--------|-------------|
| 1: Lovable scope | **(b)** Backend + Lovable updates | User updates Lovable to send `conversation_id` on every turn after turn 1. Backend implements stickiness. |
| 2: Mode-switching | **(b)** High-confidence override | Router fires every turn; only `confidence == "high"` overrides the prior mode. |
| 3: Success test | **(c)** Both unit + integration | Unit tests for stickiness logic; multi-turn smoke test for end-to-end. |

## Design

### Backend changes (this work)

In `api/main.py:/ask` handler, **before** invoking `agent.forward()`:

1. If `req.conversation_id` is present, load the most recent prior turn from `ConversationStore`.
2. Save `prior_mode` and `prior_character` for the post-routing decision.

**After** `agent.forward()` returns (router has fired):

3. Compute `effective_mode`:
   - If no `prior_mode` (turn 1 or no `conversation_id`): trust router output as today.
   - If `prior_mode` exists:
     - If `router_confidence == "high"` AND `routed_mode != prior_mode` AND `routed_mode != "none"`: honor the switch (use `routed_mode`).
     - Else: **stick** to `prior_mode`.
4. Inherit `prior_character` for `perspective_shift` mode when client sent the default.

### Why run the router on every turn (not skip it)

Cheaper would be: skip routing entirely for continuations. But that breaks Q2's high-confidence override — user couldn't switch modes mid-conversation. Acceptable cost: ~1 small LLM call per turn (already incurred today). No new spend.

### Lovable changes (your work)

1. Send `conversation_id` on every request after turn 1 (currently it's absent or rotated).
2. (Optional, polish) reflect API's `routed_mode` in the mode badge instead of what the client sent.

## Commands

```bash
# Run the focused unit tests
.venv/bin/python -m pytest tests/test_intent_routing_stickiness.py -v

# Run the full suite (must stay green)
.venv/bin/python -m pytest tests/

# Run the multi-turn smoke test against a local server
.venv/bin/python -m uvicorn api.main:app --port 8000 &
.venv/bin/python -m scripts.smoke_test_multi_turn_router

# Sanity-check intent router teacher (already exists)
.venv/bin/python -m scripts.sanity_check_teacher --mode intent_router --idx 0
```

## Project structure

Files to modify:
- `api/main.py` — stickiness logic in `/ask` handler (~30 lines added)
- `tests/test_intent_routing_stickiness.py` — **new**, ~80 lines

Files NOT to modify:
- `context_harness/intent_router.py` — the router itself stays unchanged. We only change how its output is used.
- `data/trainset_intent_router.py` — no retraining needed.
- The compiled artifact `my_profile.agent/intent_router.json` — no recompile needed.

## Code style

- Match existing conventions in `api/main.py` (snake_case, type hints, comment-on-why-not-what).
- Stickiness logic goes in a small private helper if it exceeds ~15 lines, else inline.
- New unit-test file follows the structure of existing `tests/test_*.py` (pytest, autouse fixtures where needed, parameterized for the mode-matrix).

## Testing strategy

### Unit tests (new file: `tests/test_intent_routing_stickiness.py`)

- `test_no_conversation_id_uses_router_directly` — turn 1 / one-shot path
- `test_continuation_inherits_prior_mode` — turn 2 with low-confidence "none" stays in prior mode
- `test_high_confidence_switch_overrides_stickiness` — turn 2 with high-confidence different mode
- `test_low_confidence_router_does_not_override` — turn 2 with low-confidence different mode → keeps prior
- `test_character_inherited_for_perspective_shift` — Dumbledore default carried forward
- `test_explicit_character_not_overridden` — non-default character respected even on continuation

These mock the `agent.forward()` return so they don't need a real LLM.

### Integration smoke test (`scripts/smoke_test_multi_turn_router.py`, **new**)

3-turn flow against a running server, mode=auto:
1. *"What does Hogwarts say about British boarding-school culture?"* → expect `routed_mode = open_analysis`
2. *"cool"* → expect `routed_mode = open_analysis` (stickiness fired)
3. *"tell me more"* → expect `routed_mode = open_analysis` (still sticky)

Bonus: 4th turn *"actually let's debate this"* → expect `routed_mode = debate` (high-confidence override).

Hits a real Gemini API. Skipped in CI without keys; runnable locally.

## Boundaries

### Always do
- Preserve turn-1 routing accuracy. Run existing `intent_router_metric` evals before merging to confirm no regression.
- Keep the change additive — no breaking API changes; existing callers without `conversation_id` see identical behavior.
- Trace the routing decision (`_record(turn_id, "router.decision", {...})`) so the trace viewer shows whether stickiness fired.

### Ask first
- Adding new fields to `AskResponse` (e.g., `effective_mode_source: "router" | "sticky"`) — useful for Lovable but a minor schema change.
- Touching `IntentRouterModule` itself or its trainset.
- Changing `_CHAT_MODES` membership.

### Never do
- Modify the compiled artifact JSON directly.
- Add LLM calls beyond the existing single router call per `mode=auto` request.
- Make `conversation_id` required — degrade gracefully when absent.
- Block requests when prior turn cannot be loaded (DB error etc.) — fall back to stateless routing.

## Out of scope

- Lovable frontend updates (your responsibility).
- Persisting routing telemetry beyond existing `_trace_store`.
- Confidence calibration of the router itself (e.g., changing what "high" means).
- Sticky behavior for one-shot modes (`deep_research`, `exam_grader`, `debate`, `satirical_podcast`) — only `_CHAT_MODES` (`open_analysis`, `guided_learning`, `perspective_shift`) maintain history today.

## Risks

1. **Lovable doesn't actually start sending `conversation_id`** — backend fix becomes invisible to real users. Mitigation: document the integration requirement; provide a smoke test that the user can run *after* Lovable updates.
2. **Router emits wrong confidence labels** — if "high" is over-issued, mode-switching becomes too aggressive; if under-issued, modes get stuck. Mitigation: trace logging so we can audit; backstop is the unit tests.
3. **State drift between `routed_mode` and `prior_mode` semantics** — e.g., what does "sticky" mean if `prior_mode == "none"`? Decision: if prior was off-topic, the router gets a fresh vote (no stickiness on `none` prior).

## Open questions (low priority)

- Should we surface `effective_mode_source` (`"router"` vs `"sticky"`) in `AskResponse` for the frontend to display? (Listed under "ask first" above.)
- Should turn-1-but-routed-to-`none` (off-topic on opening) be retried with a different prompt strategy, or just return the help screen as today? (Today's behavior preserved unless asked.)

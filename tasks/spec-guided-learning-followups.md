# SPEC: Smarter `guided_learning` — handle analytical follow-ups

**Status:** Confirmed, in progress
**Date:** 2026-05-15
**Branch:** `smarter-guided-learning-followups`

## Problem

Production interaction (user explicitly chose `guided_learning`):

- **Turn 1:** *"Explain blood status politics like I'm new to the series"* → good Socratic answer with Hint + Why-it-matters scaffold.
- **Turn 2:** *"is it like the racism implicitly but widely shown in the western society"* → imprecise generic answer ("both create hierarchies, foster discrimination, persecute lesser people"). Doesn't engage with "implicitly" or "Western" qualifiers.

Verified the router would have correctly classified turn 2 as `open_analysis / high` if auto mode had been used. But user pinned `guided_learning`, so router was bypassed. The mode itself owns this answer's quality.

**Root cause:** `GuidedLearningSignature` (`dspy_agent.py:183`) trains the model to emit `hint + next_question + explanation` for *every* turn, with prompts anchored on "guide the learner without revealing the answer." That framing fits *turn-1 concept-teaching* perfectly and fits poorly for *turn-2+ analytical comparisons*.

The current trainset reinforces this — zero examples with `chat_history` populated; every demo is a turn-1 concept-teaching example.

## Objective

`guided_learning` recognises analytical follow-up turns (mid-conversation, comparison/analogy phrasing) and answers in a fitting shape — direct comparison + one Socratic probe — instead of retreating to vague Hints.

**Target outcome:** turn-2 *"is it like racism in the western society"* produces a response that engages with both "implicitly" and "Western," names at least one specific HP→real-world mapping, and asks one well-targeted Socratic question.

## Acceptance criteria

1. Turn-1 concept-teaching questions still produce Hint + Why-it-matters scaffold (no regression).
2. Turn-2+ analytical comparison questions produce specific parallels in `hint`, engagement with qualifiers in `explanation`, and one targeted probe in `next_question`.
3. `socratic_metric` continues to pass for both shapes.
4. No new metric needed.

## Design

### Signature change

Update `GuidedLearningSignature` docstring to teach the turn-shape distinction.

Two response shapes:
1. **Initial concept turn** (chat_history empty or different topic): classic Socratic scaffold.
2. **Analytical follow-up turn** (chat_history shows concept introduced + current question is comparison/analogy/critique): specific parallel in `hint`, direct engagement in `explanation`, ONE targeted probe in `next_question`. NOT generic hedging.

The Signature docstring is the contract DSPy compiles into the prompt. This is the right place to encode the shape distinction, not in module logic.

### Trainset additions

Add 3 turn-2 analytical-follow-up examples with `chat_history` populated:
- **A** — the literal failing case (blood status → Western racism)
- **B** — comparison to a non-racism analogue (e.g. caste / class hierarchies)
- **C** — a "where does this break down?" critique-flavored follow-up

All three follow same shape: chat_history present + analytical question + specific direct comparison + targeted probe.

### Metric

No changes. `socratic_metric` is shape-agnostic.

### Compile

Try bootstrap first; fall back to `--max-demos 0 --max-labeled 16` if teacher drifts toward generic Hints.

## Test plan

1. Compile smoke test passes (`tests/test_compile_smoke.py`)
2. New unit test asserting metric passes on a turn-2 analytical example
3. Manual smoke: replay the actual failing conversation, verify the response shape changes

## Risks

| Risk | Mitigation |
|---|---|
| Bootstrap teacher reverts to generic-Hint pattern | Fall back to labelled-only compile |
| Turn-1 quality regresses | Smoke-test turn-1 questions; sharpen Signature wording if needed |
| Compiler picks too few new examples | Bump trainset to 6+ if first smoke fails |

## What this does NOT do

- No new mode / signature
- No routing or stickiness change
- No UI work
- No expansion to other modes

## Time estimate

~4h total: signature 0.5h, trainset 1.5h, recompile + smoke 1h, unit test 0.5h, regression check 0.5h.

# Eval coverage

Explicit documentation of what `evals/questions.py` tests and — more
importantly — what it **does not** test. Written in the spirit of
branch-coverage measurement: the known-unknowns are what makes the eval
honest. When you act on eval numbers (ship/rollback, compare model
versions, decide "this change works"), refer to this doc to know which
categories of regression your eval can actually detect.

**Maintenance cadence:** review every 4–6 weeks or whenever you:
- Add a new mode
- Change the corpus meaningfully
- Swap the LLM provider / model
- Observe a real-world failure the eval didn't catch

---

## 1. What's currently in the eval set

**File:** `evals/questions.py` · **Count:** 20 questions.

**Question kinds** (from `EvalQuestion.kind`):

| Kind | Count | What it tests | Grading method |
|---|---|---|---|
| `easy` | 8 | Single-passage retrieval | LLM judge against `reference_answer` |
| `multi` | 4 | Synthesis across ≥2 passages | LLM judge against `reference_answer` |
| `inference` | 3 | Reasoning from partial evidence | LLM judge against `reference_answer` |
| `distractor` | 5 | Refusal when answer isn't in corpus | Programmatic `_is_refusal` short-circuit |

**Corpus scope:** only the 10 canonical HP lore files in `data/hp_lore.txt`:
```
albus-dumbledore, harry-potter, hermione-granger, ron-weasley,
lord-voldemort, severus-snape, hogwarts, horcruxes,
deathly-hallows, order-of-the-phoenix
```

**Mode scope:** questions are written mode-agnostic (they all test
factual retrieval), but the `evals/eval_dspy.py` harness runs them
through a single mode at a time. In practice, the eval is
**`deep_research`-biased** — the question shapes fit deep_research
best, and the metric was designed for it.

---

## 2. What's explicitly NOT covered

### 2.1 Mode coverage gaps

Only `deep_research` is SLO-gated in CI (`slo_check.MODE_CONFIG`).

| Mode | In questions.py? | In slo_check? | Risk |
|---|---|---|---|
| `deep_research` | ✓ (all 20) | ✓ | Low — well-covered |
| `guided_learning` | ✗ | ✓ (via trainset EVALSET) | Medium — different grading shape |
| `debate` | ✗ | ✓ (via trainset EVALSET) | Medium — has its own small set |
| `open_analysis` | ✗ | ✗ | **HIGH** — no eval, no SLO gate |
| `perspective_shift` | ✗ | ✗ | **HIGH** — no eval, no SLO gate |
| `exam_grader` | ✗ | ✗ | **HIGH** — no eval, no SLO gate |
| `satirical_podcast` | ✗ | ✗ | **HIGH** — no eval, no SLO gate |

**Consequence:** a regression in any of the 4 unguarded modes ships
without detection until manual QA catches it (or a user complains).

### 2.2 Question shapes not tested

This eval does NOT include questions that require:

- **Temporal reasoning.** "Did event A happen before event B?" The HP
  corpus doesn't cleanly support this, but some questions in the real
  world do.
- **Contradictory evidence.** Questions where two corpus sources
  disagree, and the agent must acknowledge the conflict rather than pick
  a side.
- **Interpretation-dependent answers.** "Was Dumbledore a hero or a
  manipulator?" — no single correct answer. `open_analysis` mode is for
  these, but no eval set exists.
- **Stance under pushback.** User disagrees with the first answer — does
  the agent hold its ground or capitulate?
- **Multi-turn context use.** All 20 questions are single-turn. Does the
  agent correctly use `chat_history` when it's present? Not tested.
- **Ambiguous anaphora in follow-ups.** "Who killed *him*?" as a second
  turn. Tests conversation-state resolution.
- **Retrieval-ambiguity questions.** Where the top-k chunks split across
  two different correct-looking answers and the agent must pick.
- **Out-of-distribution phrasings.** Questions that use vocabulary not
  present in the corpus but refer to the same concept.
- **Corpus-adjacent distractors.** Current distractors are obviously
  outside the corpus. No distractors where the corpus *almost* answers
  the question (e.g. "In what order did Harry find the Horcruxes?" —
  corpus mentions Horcruxes but not their discovery order).
- **Length stress.** No questions that require synthesizing from 5+
  chunks simultaneously. Untested whether context budget holds up.
- **Character coverage.** Questions biased toward major characters. No
  questions about minor characters that would stress retrieval on
  lightly-represented corpus regions.
- **Non-English input.** Agent may or may not handle French / Chinese /
  Japanese questions. Untested.
- **Prompt injection.** No adversarial inputs designed to exfiltrate
  system prompts or coerce refusals.

### 2.3 Grading blind spots

- **No canon-accuracy judge.** The LLM judge grades *did the answer
  match the reference?* but doesn't independently verify *are the
  answer's claims supported by the retrieved passages?* Hallucinations
  that happen to match the reference aren't detected.
  (`metrics_llm_judge.py` is P3 scaffold — not yet wired.)
- **Citation-quality not graded.** Judge looks at the answer, not the
  citations. An answer with made-up citations that sounds plausible
  passes.
- **Refusal detector false negatives.** `_is_refusal` matches a short
  phrase list. Novel refusal phrasings the agent might adopt are missed.
- **No calibration check.** The `confidence` field from DeepResearch is
  generated but not evaluated for accuracy. High-confidence wrong
  answers aren't caught.
- **N=1 runs by default.** Single-run pass rates are noisy (~5-15% wobble
  on cloud LLMs at temp=0). `eval_dspy.py` supports `--runs N` but the
  default doesn't enforce statistical honesty.

### 2.4 Infrastructure blind spots

- **Chunking changes aren't eval-gated.** If you swap the chunker, only
  retrieval quality on these 20 questions is measured. Chunks that
  matter for questions not in the set are unmeasured.
- **Embedding-model changes aren't eval-gated.** Same story — re-embed
  with a different model, measure on the same 20 questions; effects on
  unrepresented-query regions invisible.
- **Retrieval `k` changes aren't explicitly ablated.** The eval runs at
  whatever `k` each mode's Module defaults to. No grid over `k` values.

---

## 3. Known false positives / false negatives

### 3.1 False positives (eval says pass, reality is fail)

- [ ] **TODO: add any observed cases where the eval scored an answer as
      correct but you noticed it was wrong.** Example to fill in:
      *"q20 brother wands — judge initially marked the refusal wrong
      because judge used external HP knowledge. Fixed by kind-aware
      short-circuit on 2026-04-XX."*

### 3.2 False negatives (eval says fail, reality is pass)

- [ ] **TODO: add cases where the eval marked an answer wrong but it
      was actually fine.** Often due to reference-answer phrasing
      being too narrow.

### 3.3 Flaky questions

Questions that give different verdicts across runs. Symptom: pass rate
varies by >10% run-to-run with temperature=0.

- [ ] **TODO: list flaky question IDs + root cause if known.**

---

## 4. Questions added from observed failures

This is the highest-leverage section to fill in over time. Each row
should be a real failure you observed → a question added to catch it.

| Date | Failure observed | Question added | Question ID |
|---|---|---|---|
| 2026-04-23 | Friend asked "What is the origin of Quidditch game?" via Lovable demo. Agent correctly refused (no Quidditch doc in corpus), but user perceived it as a failure — "basic HP question got I-don't-know." Corpus gap on a high-profile topic. | Added `doc_id: quidditch` paragraph to `data/hp_lore.txt` (origin, rules, positions, World Cup, famous teams). Added eval question covering origin + rules. | q21 |

This section turns your eval from "hand-written questions" into
"regression log with question-shaped entries." Fill in every time you
fix a bug the eval didn't catch.

---

## 5. Diagnostic-vs-regression split

Current 20 questions are predominantly **regression guards** (catch
catastrophic failure). Few are **diagnostic** (discriminate between
two plausible versions of the system).

Target distribution for a mature eval (per the diagnostic-question
framework):

- 3–5 sanity-check easy questions (regression guards)
- 6–8 difficulty-gradient questions (L1→L4 ladders on target facts)
- 3–5 oracle-disagreement questions (mined from runs where judge and
  retrieval disagreed)
- 3–5 failure-mode-targeted questions (each probes a specific pipeline
  component)
- 2–3 corpus-adjacent distractors (where corpus almost-but-not-quite
  answers)

**Current distribution:** 8 easy + 4 multi + 3 inference + 5 distractor
= mostly regression guards, minimal diagnostic coverage.

**Gap:** no questions exist specifically to discriminate between, e.g.,
compiled vs uncompiled modules, or two different chunking strategies.
Adding those is the fastest way to make the eval actually decision-useful.

---

## 6. Planned expansions

Priority order (fill in / adjust):

- [ ] **Oracle-disagreement mining** — scan existing eval JSON output
      for rows where `correct != retrieval_hit`. Each is a candidate
      diagnostic question.
- [ ] **Multi-turn conversation tests** — at least 5 questions
      structured as a 3-turn chat to cover the 3 chat modes
      (`open_analysis`, `guided_learning`, `perspective_shift`).
- [ ] **Per-mode eval sets** — dedicated 10-question eval sets for
      `open_analysis`, `perspective_shift`, `exam_grader`,
      `satirical_podcast` so those modes have SLO gates.
- [ ] **Canon-accuracy judge wiring** — finish `metrics_llm_judge.py`
      integration in `eval_dspy.py` to catch unsupported claims.
- [ ] **Snapshot canonical traces** — 3–5 highest-diagnostic questions
      saved as frozen JSON transcripts; diff-on-run catches "accuracy
      held but tool-call pattern changed" drift.
- [ ] **Corpus-adjacent distractors** — add 3 distractors where the
      corpus has fragments but not the specific answer.

---

## 7. What this doc is not

- Not a spec of "questions I wish we had." Only questions you've
  committed to writing + running go here.
- Not a design doc for new modes. Separate file.
- Not a grading-algorithm reference. See `context_harness/metrics.py`
  for the metric functions.

---

## Closing principle

> **The value of writing this doc is not in the answers — it's in the
> forced honesty about what the eval does and doesn't see.**
>
> A 90% pass rate on a set of questions that covers 40% of possible
> failure modes tells you less than a 70% pass rate on a set that
> covers 90%. This doc is how you measure the denominator.

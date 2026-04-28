"""
DSPy Metrics
============
Metric functions for the DSPy optimizer. A metric takes (example, prediction,
trace=None) and returns True/False (or a float score). DSPy uses the metric to:

  1. Select which bootstrapped traces become few-shot demonstrations
  2. Score the program during evaluation

Design principle: metrics encode "what good looks like" for each mode.
The optimizer's only job is to maximise the metric, so a weak metric
produces a weak agent. Put effort here.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Deep Research — citation accuracy
# ---------------------------------------------------------------------------

def deep_research_metric(example, pred, trace=None) -> bool:
    """
    True when at least half of the expected citations appear in the prediction.
    This is deliberately lenient — citation sets can legitimately vary.

    Example labels: `example.citations = "horcruxes lord-voldemort"`
    Pred output:    `pred.citations   = "horcruxes lord-voldemort harry-potter"`
    """
    expected = _tokens(getattr(example, "citations", ""))
    actual   = _tokens(getattr(pred, "citations", ""))
    if not expected:
        return True   # no constraint labelled
    overlap = len(expected & actual)
    return overlap >= (len(expected) + 1) // 2


def deep_research_strict_metric(example, pred, trace=None) -> bool:
    """Strict variant: every expected citation must appear."""
    expected = _tokens(getattr(example, "citations", ""))
    actual   = _tokens(getattr(pred, "citations", ""))
    return expected.issubset(actual) if expected else True


# Refusal phrases — signals the agent declined to answer. These are chosen
# to minimise false positives on legitimate factual answers that happen to
# share vocabulary (e.g. an answer saying the Elder Wand "was not available
# to Harry before 1998" is NOT a refusal). Phrases are anchored to wording
# that only appears when the agent is framing the RESPONSE ITSELF as
# unavailable, not a canon event.
_REFUSAL_PHRASES = (
    "context does not contain",
    "text does not contain",
    "i don't know",
    "information is not available",
    "answer is not available",
    "cannot answer",
    "no information about",
    "not mentioned in the",
    "i'm sorry, but the provided",
    "the provided context does not",
)


def _is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def deep_research_balanced_metric(example, pred, trace=None) -> bool:
    """
    Balanced metric that checks both precision (citations) and recall (no
    false refusals). Penalises the agent for refusing to answer non-distractor
    questions, even if citation accuracy is perfect.

    True when:
      1. Citation overlap >= 50% (or no citations expected)
      2. Agent did NOT refuse on a non-distractor question
      3. Answer is non-trivial (>30 chars)
    """
    answer = getattr(pred, "answer", "") or ""
    kind = getattr(example, "kind", "easy")

    # Citation check (precision)
    expected = _tokens(getattr(example, "citations", ""))
    actual = _tokens(getattr(pred, "citations", ""))
    if expected:
        overlap = len(expected & actual)
        if overlap < (len(expected) + 1) // 2:
            return False

    # Refusal penalty (recall) — only for non-distractor questions
    if _is_refusal(answer) and kind != "distractor":
        return False

    # Trivially short answers aren't useful
    if len(answer) < 30:
        return False

    return True


def deep_research_score(example, pred, trace=None) -> float:
    """
    Continuous 0.0–1.0 score for ranking. Gives the optimizer more signal
    than binary pass/fail.

    Components:
      0.4 — citation overlap ratio
      0.3 — non-refusal on answerable questions (or refusal on distractors)
      0.2 — answer length (30-500 chars sweet spot)
      0.1 — confidence is stated
    """
    score = 0.0
    answer = getattr(pred, "answer", "") or ""
    kind = getattr(example, "kind", "easy")

    # Citation overlap (0.4)
    expected = _tokens(getattr(example, "citations", ""))
    actual = _tokens(getattr(pred, "citations", ""))
    if expected:
        overlap = len(expected & actual) / len(expected)
        score += 0.4 * overlap
    else:
        score += 0.4  # no citations expected — full marks

    # Refusal appropriateness (0.3)
    refused = _is_refusal(answer)
    if kind == "distractor" and refused:
        score += 0.3  # correctly refused
    elif kind != "distractor" and not refused:
        score += 0.3  # correctly answered
    # else: wrong behavior, 0 points

    # Answer length (0.2)
    if 30 <= len(answer) <= 500:
        score += 0.2
    elif len(answer) > 500:
        score += 0.1  # verbose but still useful

    # Confidence stated (0.1)
    confidence = getattr(pred, "confidence", "") or ""
    if confidence.strip().lower() in ("low", "medium", "high"):
        score += 0.1

    return score


# ---------------------------------------------------------------------------
# Guided Learning — Socratic score
# ---------------------------------------------------------------------------

# Words that usually signal "this sentence contains the answer"
_SPOILER_PHRASES = (
    "the answer is",
    "this is because",
    "voldemort is",
    "dumbledore is",
    "a horcrux is",
    "the deathly hallows are",
    "the elder wand is",
    "the invisibility cloak is",
    "the resurrection stone is",
)


def socratic_metric(example, pred, trace=None) -> bool:
    """
    True when:
      1. The hint does NOT contain a direct-answer spoiler phrase
      2. next_question ends with a question mark (active probing)
      3. explanation discusses concept, not mechanism (no spoiler phrases)
    """
    hint          = (getattr(pred, "hint", "") or "").lower()
    next_question = (getattr(pred, "next_question", "") or "").strip()
    explanation   = (getattr(pred, "explanation", "") or "").lower()

    if _contains_spoiler(hint):
        return False
    if _contains_spoiler(explanation):
        return False
    if not next_question.endswith("?"):
        return False
    if len(hint) < 20:   # trivially short hints aren't useful
        return False
    return True


def socratic_score(example, pred, trace=None) -> float:
    """Continuous variant — returns a 0.0 → 1.0 score for ranking."""
    score = 0.0
    hint          = (getattr(pred, "hint", "") or "").lower()
    next_question = (getattr(pred, "next_question", "") or "").strip()
    explanation   = (getattr(pred, "explanation", "") or "").lower()

    if hint and not _contains_spoiler(hint):
        score += 0.4
    if next_question.endswith("?"):
        score += 0.3
    if explanation and not _contains_spoiler(explanation):
        score += 0.2
    if 30 <= len(hint) <= 400:   # meaningful length
        score += 0.1
    return score


# ---------------------------------------------------------------------------
# Perspective Shift — character principle applied to real world
# ---------------------------------------------------------------------------

# Greeting phrases that indicate cold-start voice on a turn-2+ reply.
# Regression target (2026-04-21 Luna bug): the LLM would open with
# "Oh, hello there!" on turn 3 of a continuing conversation, ignoring
# the chat_history. Compile-time metric rejects these so BootstrapFewShot
# never promotes a greeting-opener response into the demo pool.
_GREETING_OPENERS = (
    "oh, hello",
    "hi there",
    "hello there",
    "hello,",
    "hi,",
    "welcome",
    "nice to meet",
)


def _opens_with_greeting(text: str) -> bool:
    """True if the first non-whitespace token is a cold-start greeting."""
    return any(text.lstrip().lower().startswith(g) for g in _GREETING_OPENERS)


# Patterns that signal the user is asking for reflection / summary.
_REFLECTION_TRIGGERS = (
    "summarize",
    "summarise",
    "what have i told you",
    "what did i tell you",
    "what did i say",
    "what have i said",
    "reflect back",
    "recap",
    "my situation",
    "what do you know about me",
)

# Common words to ignore when measuring content overlap.
_STOPWORDS = frozenset(
    "i me my you your a an the is am are was were be been being "
    "do does did have has had it its in on at to for of and or but "
    "not no so if by with from that this these those than then "
    "about just really very also too all both each "
    "what when where who how which why would could should can may "
    "will shall might must well most more some any our her him his "
    "them they she he we us them into out up even still yet".split()
)


def _is_reflection_request(scenario: str) -> bool:
    """True when the user's scenario looks like a reflection / summary ask."""
    lower = scenario.lower()
    return any(trigger in lower for trigger in _REFLECTION_TRIGGERS)


def _content_words(text: str) -> set[str]:
    """Extract lowercased non-stop content words (3+ chars) from text."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def perspective_shift_metric(example, pred, trace=None) -> bool:
    """
    True when:
      1. character_principle is grounded (>= 50 chars, specific)
      2. applied_insight is actionable (>= 80 chars, not just "be brave")
      3. reasoning bridges character to scenario (>= 50 chars)
      4. character_response is substantive (>= 100 chars) — the synthesis
         field added in PR #21; without checking it, the metric rewards
         good analytical fields even when the user-facing voice is bad
      5. citations present (character grounded in corpus)
      6. MULTI-TURN GUARD: if example.chat_history is non-empty, the
         character_response must NOT open with a greeting. Enforces the
         no-cold-start rule at compile time so demos selected by
         BootstrapFewShot don't reinforce the bug the 2026-04-21 Luna
         observation surfaced.
    """
    principle = getattr(pred, "character_principle", "") or ""
    insight = getattr(pred, "applied_insight", "") or ""
    reasoning = getattr(pred, "reasoning", "") or ""
    char_response = getattr(pred, "character_response", "") or ""
    citations = getattr(pred, "citations", "") or ""

    if len(principle) < 50:
        return False
    if len(insight) < 80:
        return False
    if len(reasoning) < 50:
        return False
    if len(char_response.strip()) < 100:
        return False
    if not citations or citations.strip() == "none":
        return False

    # Multi-turn greeting guard
    chat_history = getattr(example, "chat_history", "") or ""
    if chat_history.strip() and _opens_with_greeting(char_response):
        return False

    # Reflection guard: when the user explicitly asks for a summary /
    # reflection, the character_response must echo back content from the
    # chat_history — at least 3 overlapping content words. A response that
    # ignores the user's prior turns entirely fails.
    scenario = getattr(example, "scenario", "") or ""
    if chat_history.strip() and _is_reflection_request(scenario):
        history_words = _content_words(chat_history)
        response_words = _content_words(char_response)
        if len(history_words & response_words) < 3:
            return False

    return True


# ---------------------------------------------------------------------------
# Open Analysis — quality of analytical response
# ---------------------------------------------------------------------------

def open_analysis_metric(example, pred, trace=None) -> bool:
    """
    True when:
      1. analysis is substantial (>100 chars)
      2. own_reasoning is non-empty (the LLM actually went beyond the corpus)
      3. corpus_facts is non-empty (the LLM grounded in retrieved data)
    """
    analysis = getattr(pred, "analysis", "") or ""
    own_reasoning = getattr(pred, "own_reasoning", "") or ""
    corpus_facts = getattr(pred, "corpus_facts", "") or ""

    if len(analysis) < 100:
        return False
    if len(own_reasoning) < 30:
        return False
    if len(corpus_facts) < 20:
        return False
    return True


# ---------------------------------------------------------------------------
# Exam Grader — grading accuracy
# ---------------------------------------------------------------------------

def exam_grader_metric(example, pred, trace=None) -> bool:
    """
    True when:
      1. The predicted score is within 15 points of the expected score
      2. is_passing agrees with the expected value
      3. critique is non-trivial (>20 chars)
    """
    expected_score = int(getattr(example, "expected_score", 0))
    expected_passing = getattr(example, "expected_passing", None)

    pred_score = int(getattr(pred, "score", 0))
    pred_passing = getattr(pred, "is_passing", None)
    critique = getattr(pred, "critique", "") or ""

    if abs(pred_score - expected_score) > 15:
        return False
    if expected_passing is not None and bool(pred_passing) != bool(expected_passing):
        return False
    if len(critique) < 20:
        return False
    return True


# ---------------------------------------------------------------------------
# Satirical Podcast — transcript structure + canon grounding
# ---------------------------------------------------------------------------

def satirical_podcast_metric(example, pred, trace=None) -> bool:
    """
    True when all of:
      1. transcript is substantial (≥ 150 chars)
      2. transcript contains at least 3 dialogue lines formatted as "Name: text"
      3. comedic_tension is non-trivial (≥ 20 chars)
      4. citations field is non-empty (at least one canon doc referenced)

    Note: we deliberately do NOT check funniness — that requires an LLM judge.
    This metric guards structure and canon grounding only.
    """
    transcript       = (getattr(pred, "transcript",       "") or "").strip()
    comedic_tension  = (getattr(pred, "comedic_tension",  "") or "").strip()
    citations        = (getattr(pred, "citations",        "") or "").strip()

    if len(transcript) < 150:
        return False
    if len(comedic_tension) < 20:
        return False
    if not citations or citations.lower() == "none":
        return False
    # At least 3 "Speaker: dialogue" lines
    dialogue_lines = re.findall(r"^\s*\w[\w\s]*:\s+\S", transcript, re.MULTILINE)
    if len(dialogue_lines) < 3:
        return False
    return True


def satirical_podcast_score(example, pred, trace=None) -> float:
    """Continuous variant — 0.0 → 1.0 for optimizer ranking."""
    score = 0.0
    transcript      = (getattr(pred, "transcript",      "") or "").strip()
    comedic_tension = (getattr(pred, "comedic_tension", "") or "").strip()
    citations       = (getattr(pred, "citations",       "") or "").strip()

    dialogue_lines = re.findall(r"^\s*\w[\w\s]*:\s+\S", transcript, re.MULTILINE)

    if len(transcript)      >= 150: score += 0.3
    if len(dialogue_lines)  >= 3:   score += 0.3
    if len(comedic_tension) >= 20:  score += 0.2
    if citations and citations.lower() != "none":
        score += 0.2

    return score


# ---------------------------------------------------------------------------
# Debate — balanced argument quality
# ---------------------------------------------------------------------------

def debate_metric(example, pred, trace=None) -> bool:
    """
    True when all of:
      1. arguments_for is non-empty (≥ 20 chars)
      2. arguments_against is non-empty (≥ 20 chars)
      3. verdict is non-empty (≥ 10 chars)
      4. At least one expected citation appears in the prediction
    """
    args_for     = (getattr(pred, "arguments_for",     "") or "").strip()
    args_against = (getattr(pred, "arguments_against", "") or "").strip()
    verdict      = (getattr(pred, "verdict",           "") or "").strip()
    citations    = (getattr(pred, "citations",         "") or "")

    if len(args_for)     < 20: return False
    if len(args_against) < 20: return False
    if len(verdict)      < 10: return False

    expected = _tokens(getattr(example, "citations", ""))
    if expected and not (expected & _tokens(citations)):
        return False

    return True


def debate_score(example, pred, trace=None) -> float:
    """Continuous variant — 0.0 → 1.0 for optimizer ranking."""
    score = 0.0
    args_for     = (getattr(pred, "arguments_for",     "") or "").strip()
    args_against = (getattr(pred, "arguments_against", "") or "").strip()
    verdict      = (getattr(pred, "verdict",           "") or "").strip()
    citations    = _tokens(getattr(pred, "citations",  ""))

    if len(args_for)     >= 20: score += 0.3
    if len(args_against) >= 20: score += 0.3
    if len(verdict)      >= 10: score += 0.2

    expected = _tokens(getattr(example, "citations", ""))
    if expected and expected & citations:
        score += 0.2

    return score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens(s: str) -> set[str]:
    """Split a space-separated citation string into a normalised token set."""
    return {t.strip().lower() for t in (s or "").split() if t.strip()}


def _contains_spoiler(text: str) -> bool:
    return any(phrase in text for phrase in _SPOILER_PHRASES)


# ---------------------------------------------------------------------------
# Intent Router — mode classification accuracy
# ---------------------------------------------------------------------------

_ROUTER_VALID_MODES = {
    "deep_research", "guided_learning", "exam_grader", "open_analysis",
    "perspective_shift", "debate", "satirical_podcast", "none",
}


def intent_router_metric(example, pred, trace=None) -> bool:
    """
    True when:
      1. Predicted mode matches expected mode (exact, after normalization)
      2. Required kwargs keys are present (doesn't check values)
      3. kwargs_json is valid JSON
    """
    import json as _json

    expected_mode = (getattr(example, "mode", "") or "").strip().lower()
    pred_mode = (getattr(pred, "mode", "") or "").strip().lower().replace(" ", "_").replace("-", "_")

    # Mode must match
    if pred_mode != expected_mode:
        return False

    # kwargs_json must be valid JSON
    pred_kwargs_str = (getattr(pred, "kwargs_json", "") or "").strip()
    try:
        pred_kwargs = _json.loads(pred_kwargs_str)
        if not isinstance(pred_kwargs, dict):
            return False
    except (ValueError, TypeError):
        return False

    # Required kwargs keys must be present
    expected_kwargs_str = (getattr(example, "kwargs_json", "") or "").strip()
    try:
        expected_kwargs = _json.loads(expected_kwargs_str)
        if isinstance(expected_kwargs, dict):
            for key in expected_kwargs:
                if key not in pred_kwargs:
                    return False
    except (ValueError, TypeError):
        pass  # no expected kwargs to check

    return True


def intent_router_score(example, pred, trace=None) -> float:
    """
    Continuous 0.0–1.0 score for ranking.

    Components:
      0.6 — mode match
      0.2 — kwargs keys correct
      0.2 — valid confidence value
    """
    import json as _json

    score = 0.0

    expected_mode = (getattr(example, "mode", "") or "").strip().lower()
    pred_mode = (getattr(pred, "mode", "") or "").strip().lower().replace(" ", "_").replace("-", "_")

    # Mode match (0.6)
    if pred_mode == expected_mode:
        score += 0.6

    # kwargs correctness (0.2)
    pred_kwargs_str = (getattr(pred, "kwargs_json", "") or "").strip()
    try:
        pred_kwargs = _json.loads(pred_kwargs_str)
        if isinstance(pred_kwargs, dict):
            expected_kwargs_str = (getattr(example, "kwargs_json", "") or "").strip()
            try:
                expected_kwargs = _json.loads(expected_kwargs_str)
                if isinstance(expected_kwargs, dict):
                    if not expected_kwargs or all(k in pred_kwargs for k in expected_kwargs):
                        score += 0.2
                else:
                    score += 0.2  # no structured kwargs expected
            except (ValueError, TypeError):
                score += 0.2  # no expected kwargs to check
    except (ValueError, TypeError):
        pass  # invalid JSON — 0 points

    # Valid confidence (0.2)
    confidence = (getattr(pred, "confidence", "") or "").strip().lower()
    if confidence in ("low", "medium", "high"):
        score += 0.2

    return score

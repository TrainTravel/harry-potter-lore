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

def perspective_shift_metric(example, pred, trace=None) -> bool:
    """
    True when:
      1. character_principle is grounded (mentions specific events, not generic)
      2. applied_insight is actionable (>80 chars, not just "be brave")
      3. reasoning bridges character to scenario (>50 chars)
      4. citations present (character grounded in corpus)
    """
    principle = getattr(pred, "character_principle", "") or ""
    insight = getattr(pred, "applied_insight", "") or ""
    reasoning = getattr(pred, "reasoning", "") or ""
    citations = getattr(pred, "citations", "") or ""

    if len(principle) < 50:
        return False
    if len(insight) < 80:
        return False
    if len(reasoning) < 50:
        return False
    if not citations or citations.strip() == "none":
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

"""
Intent Router — LLM-based mode classification
===============================================
Classifies a user message into one of the agent's modes and extracts
mode-specific kwargs (character, student_answer, modern_angle, etc.).

Design:
  - Pure classification — no retrieval, no pipeline dependency.
  - One extra Flash-Lite call per request (~0.3s, ~$0.00002).
  - Mode descriptions live in the Signature docstring so they compile
    into the prompt and travel with the JSON artifact — no separate
    config file to keep in sync.
  - ``kwargs_json`` is a string (DSPy outputs are strings). Parsed
    defensively after prediction; falls back to ``{}``.

Usage:
    router = IntentRouterModule()
    pred = router.forward(user_message="Sort me into a house!")
    result = parse_router_output(pred)
    # result = {"mode": "perspective_shift", "kwargs": {"character": "Sorting Hat"}, ...}
"""

from __future__ import annotations

import json
from typing import Any, Dict, Set

import dspy


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

class IntentRouterSignature(dspy.Signature):
    """Classify a user's message into the best-matching Harry Potter agent mode
    and extract any parameters the mode needs.

    Available modes:

    deep_research — broad factual questions about HP lore, history, spells,
        creatures, events. Example: "What are the properties of the Elder Wand?"

    guided_learning — Socratic tutoring. The user wants to learn, not just get
        an answer. Example: "Help me understand how Horcruxes work."

    exam_grader — the user provides a student answer to be graded against canon.
        Extract the exam question and student_answer into kwargs_json.
        Example: "Grade this answer: Q: What are the Deathly Hallows? A: Three
        powerful objects created by Death."

    open_analysis — analytical, interpretive, or comparative questions that go
        beyond factual retrieval. Example: "Why did Snape become the person he
        became?"

    perspective_shift — ANY personal life situation, emotional question,
        decision the user is facing, or request for advice — regardless of
        whether the user names an HP character or mentions HP at all. Grief,
        relationships, career doubts, family conflict, mental-health worries,
        self-doubt, identity questions, "what do you think about [a person /
        a situation]", "how do I cope with X", "I'm struggling with Y" all
        belong here. If the user names a character, extract them into
        kwargs_json; if not, leave kwargs empty — the product will ask the
        user to pick a character.
        Examples:
        - "What would Dumbledore say about my career change?"
        - "I'm going through a hard time."
        - "How do I deal with imposter syndrome?"
        - "What do you think about a mother who lost her sons?"
        - "Sort me into a house" (character="Sorting Hat")

    debate — the user states a debatable claim about HP canon and wants both
        sides argued. Example: "Was Snape really a hero?"

    satirical_podcast — the user wants a comedic take on an HP topic through a
        modern lens. Extract the modern_angle into kwargs_json if stated.
        Example: "Do a podcast about Quidditch as an extreme sport industry."

    none — reserve for messages that are clearly NEITHER about Harry Potter
        NOR a personal life situation. Pure off-topic factual queries
        (weather, math, stock prices, news, non-HP fiction) belong here.
        Plain greetings with no other content ("hi", "hello") also belong
        here. When in doubt between "none" and "perspective_shift" on an
        emotional or life-situation message, prefer "perspective_shift" —
        the product is designed to engage with these queries through a
        character's voice.

    Rules:
    - Choose exactly ONE mode.
    - For perspective_shift: extract character into kwargs_json IF the user
      names one. Otherwise leave kwargs empty ({}). If the user says "sort
      me" or "put me into a house", use character="Sorting Hat".
    - For exam_grader: extract student_answer into kwargs_json.
    - For satirical_podcast: extract modern_angle into kwargs_json if stated.
    - kwargs_json must be valid JSON or an empty object "{}".
    - confidence: "high" when the intent is unambiguous, "medium" when 2 modes
      could apply, "low" when guessing.
    """

    user_message: str = dspy.InputField(desc="the user's raw message to classify")

    mode: str = dspy.OutputField(
        desc="one of: deep_research, guided_learning, exam_grader, "
             "open_analysis, perspective_shift, debate, satirical_podcast, none"
    )
    kwargs_json: str = dspy.OutputField(
        desc='JSON object with extracted parameters for the chosen mode, '
             'e.g. {"character": "Dumbledore"} or "{}" if none needed'
    )
    confidence: str = dspy.OutputField(
        desc="one of: low, medium, high"
    )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

VALID_MODES: Set[str] = {
    "deep_research", "guided_learning", "exam_grader", "open_analysis",
    "perspective_shift", "debate", "satirical_podcast", "none",
}


class IntentRouterModule(dspy.Module):
    """Classify user intent into a mode. No pipeline dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(IntentRouterSignature)

    def forward(self, user_message: str = "") -> dspy.Prediction:
        return self.predict(user_message=user_message)


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def parse_router_output(pred: dspy.Prediction) -> Dict[str, Any]:
    """Parse and validate the router's prediction.

    Returns:
        {
            "mode": str,         # validated mode or "none"
            "kwargs": dict,      # parsed kwargs or {}
            "confidence": str,   # normalized confidence
            "raw_mode": str,     # original mode string before normalization
        }
    """
    raw_mode = (getattr(pred, "mode", "") or "").strip().lower()
    # Normalize common variations
    mode = raw_mode.replace(" ", "_").replace("-", "_")
    if mode not in VALID_MODES:
        mode = "none"

    # Parse kwargs_json defensively
    kwargs_json_str = (getattr(pred, "kwargs_json", "") or "").strip()
    try:
        kwargs = json.loads(kwargs_json_str)
        if not isinstance(kwargs, dict):
            kwargs = {}
    except (json.JSONDecodeError, TypeError):
        kwargs = {}

    # Normalize confidence
    raw_confidence = (getattr(pred, "confidence", "") or "").strip().lower()
    confidence = raw_confidence if raw_confidence in ("low", "medium", "high") else "low"

    return {
        "mode": mode,
        "kwargs": kwargs,
        "confidence": confidence,
        "raw_mode": raw_mode,
    }

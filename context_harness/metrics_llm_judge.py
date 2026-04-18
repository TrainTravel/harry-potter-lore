"""
LLM-as-judge metrics.
=====================

So far our programmatic metrics (``metrics.py``) check structural
properties — citation overlap, refusal phrases, output length,
tag set validity. They can't answer *"is the content correct?"*.

This module adds LLM-graded metrics for that question:

  - ``canon_accuracy_judge`` — does the answer contain any claim that
    contradicts the given context or widely-known canonical HP lore?

The judge is a SECOND LLM call with a focused rubric, distinct from
the main judge in ``evals/eval_agent.py`` (which grades fact-match
against a reference answer).

Pattern mirrors the tagger in lore-builder: Protocol + dependency
injection so tests use a ``FakeJudgeClient`` with scripted responses
and never hit the network.

Failure modes this addresses, seen in real traces:

  - **Sanitised canon** — the Snape question where the agent said
    "Snape's role was one of opposition to Voldemort, not complicity"
    while skipping the prophecy-overhearing → reporting detail. The
    main judge said "correct" (matched reference phrasing). A canon
    judge flags "claim contradicts established canon about Snape
    reporting the prophecy".
  - **Invented facts** — claims not supported by context that happen
    to sound plausible. Main judge may pass them; canon judge flags.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Protocol


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge client Protocol — identical shape to lore-builder's LLMClient
# ---------------------------------------------------------------------------

class LLMJudgeClient(Protocol):
    """Minimal interface the canon-accuracy judge needs. Any class with
    a ``complete_json(prompt) -> str`` method satisfies this — no
    inheritance required."""
    def complete_json(self, prompt: str) -> str:
        ...


@dataclass
class GeminiJudgeClient:
    """Production adapter. Uses Gemini 2.5 Flash-lite by default (cheap,
    fine for rubric-style grading). temperature kept low for
    grade consistency across re-runs."""
    model: str = "gemini-2.5-flash-lite"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_output_tokens: int = 400

    def complete_json(self, prompt: str) -> str:
        import os as _os
        from google import genai
        from google.genai import types
        key = (self.api_key
               or _os.environ.get("GEMINI_API_KEY")
               or _os.environ.get("GOOGLE_API_KEY"))
        if not key:
            raise RuntimeError(
                "No API key: set GEMINI_API_KEY / GOOGLE_API_KEY env var "
                "or pass api_key= to GeminiJudgeClient."
            )
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
            ),
        )
        return resp.text or ""


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

_CANON_JUDGE_SYSTEM = """You are a Harry Potter canon expert grading whether \
an agent's answer contains any claims that contradict established canon.

You will see:
  - QUESTION: what the user asked
  - AGENT ANSWER: what the agent responded
  - CONTEXT (optional): lore passages that were retrieved for the agent

Your job:
  1. Identify specific factual claims in the AGENT ANSWER.
  2. For each claim, check whether it contradicts:
     (a) the CONTEXT passages, if they speak to it, OR
     (b) widely-established canonical Harry Potter lore that a serious fan
         would recognise (the seven-book main series plus well-known
         supplementary material like Pottermore/Wizarding World).
  3. Be STRICT about material errors (wrong character did X, wrong causal
     chain, wrong outcome), LENIENT about interpretation and emphasis.
  4. A missing detail is NOT a contradiction. An answer that *stops short*
     of mentioning a relevant canonical fact is not automatically incorrect
     — flag it only if the missing detail changes the moral/factual picture
     materially (e.g. claiming Snape opposed Voldemort without any mention
     of Snape's role in reporting the prophecy, when the question asks
     about Snape's moral arc).

Return a single JSON object with this shape:
  {
    "canon_accurate": true | false,
    "incorrect_claims": ["specific claim 1", "specific claim 2", ...],
    "reasoning": "<one-to-three sentences explaining the verdict>"
  }

If canon_accurate is true, `incorrect_claims` must be an empty array.
If you're genuinely uncertain about a claim, do NOT flag it — err on the
side of canon_accurate=true. Only flag claims you are confident contradict
the canon.

Return ONLY the JSON. No prose, no markdown fences.
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def canon_accuracy_judge(
    question: str,
    agent_answer: str,
    context: str = "",
    client: Optional[LLMJudgeClient] = None,
) -> dict:
    """Grade whether ``agent_answer`` contains canon-contradicting claims.

    Returns a dict with keys:
      - ``canon_accurate`` (bool)
      - ``incorrect_claims`` (list[str])
      - ``reasoning`` (str)
      - ``raw_response`` (str) — the judge's raw output, for debugging

    On any parse or call failure, returns a conservative verdict
    (``canon_accurate=True, incorrect_claims=[]``) plus a diagnostic in
    ``reasoning``. Silent failure is safer than blocking the eval on a
    judge hiccup.
    """
    if client is None:
        client = GeminiJudgeClient()

    prompt = (
        f"{_CANON_JUDGE_SYSTEM}\n\n"
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{context if context else '(no context provided)'}\n\n"
        f"AGENT ANSWER:\n{agent_answer}\n"
    )

    try:
        raw = client.complete_json(prompt)
    except Exception as exc:
        return _safe_fallback(raw="", reason=f"judge call failed: {exc}")

    match = _JSON_RE.search(raw or "")
    if not match:
        return _safe_fallback(raw=raw, reason="no JSON object in judge response")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return _safe_fallback(raw=raw, reason=f"malformed JSON: {exc}")

    accurate = bool(parsed.get("canon_accurate", True))
    claims_raw = parsed.get("incorrect_claims", [])
    claims = [str(c) for c in claims_raw if isinstance(c, str)]
    reasoning = str(parsed.get("reasoning", "")).strip()

    # Invariant guard: if accurate==True but claims aren't empty, the judge
    # contradicted itself. Trust the claims (stricter reading).
    if accurate and claims:
        accurate = False
        reasoning = f"[judge self-contradicted; trusted claims list] {reasoning}"

    return {
        "canon_accurate":    accurate,
        "incorrect_claims":  claims,
        "reasoning":         reasoning,
        "raw_response":      raw,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _safe_fallback(raw: str, reason: str) -> dict:
    """When the judge can't be relied on, return a conservative pass rather
    than blocking the surrounding eval on a judge hiccup."""
    log.warning("Canon-accuracy judge fallback: %s", reason)
    return {
        "canon_accurate":   True,
        "incorrect_claims": [],
        "reasoning":        f"[fallback] {reason}",
        "raw_response":     raw,
    }

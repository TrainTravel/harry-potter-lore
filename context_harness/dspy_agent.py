"""
DSPy Agent System
=================
Two-mode agent backed by DSPy for automated prompt optimisation and ChromaDB
for vector memory.

Design (from design summary):
  - Signatures declare input/output contracts. Model weights never change.
  - Modules compose Signatures with ChromaDB retrieval.
  - Compiled JSON artifacts store the optimised few-shots + instructions
    produced by a DSPy optimizer. Recompile when the model or training set changes.
  - manifest.json guards against embedding model drift (backfill guard).

Modes
-----
  deep_research    — broad retrieval (k=10), outputs: answer, citations,
                     confidence, gaps
  guided_learning  — narrow retrieval (k=3) filtered to past attempts,
                     Socratic outputs: hint, next_question, explanation

Usage
-----
    from context_harness.dspy_agent import DSPyAgent
    from context_harness.ingest_lore import build_pipeline

    pipeline = build_pipeline(persist=False)
    agent = DSPyAgent(pipeline)
    result = agent.forward("deep_research", "Who created the Deathly Hallows?")
    print(result.answer, result.citations)

Export / import
---------------
    agent.save("my_profile.agent")
    # → my_profile.agent/deep_research.json
    # → my_profile.agent/guided_learning.json
    # → my_profile.agent/manifest.json

    agent2 = DSPyAgent(pipeline, export_dir="my_profile.agent")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import dspy

from .intent_router import IntentRouterModule, parse_router_output
from .rag_pipeline import RAGPipeline
from .retrieval_tools import build_retrieval_registry


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

class DeepResearchSignature(dspy.Signature):
    """Answer a question with depth and precision using retrieved lore context.
    Cite your sources, rate your confidence, and flag information gaps."""

    question: str = dspy.InputField(desc="the research question")
    context: str  = dspy.InputField(desc="retrieved lore passages, each prefixed [doc_id]")

    answer:     str = dspy.OutputField(desc="comprehensive answer drawn from context")
    citations:  str = dspy.OutputField(desc="space-separated doc_ids referenced in the answer")
    confidence: str = dspy.OutputField(desc="one of: low / medium / high")
    gaps:       str = dspy.OutputField(desc="aspects the context does not cover, or 'none'")


class QueryPlanSignature(dspy.Signature):
    """Break a complex HP lore question into 2-3 targeted retrieval sub-queries.
    Output a JSON array of tool calls. Available tools:
      - vector_search(query: str, k: int 1-15) — broad semantic search
      - character_search(character: str, query: str, k: int 1-10) — scoped to a character slug
      - topic_search(topic: str, query: str, k: int 1-10) — topic-prefixed search
    Use character_search when a specific character is central.
    Use topic_search for artefacts, institutions, or events.
    Use vector_search for general or cross-cutting questions."""

    question: str = dspy.InputField(desc="the HP lore research question to plan retrieval for")
    tool_calls: str = dspy.OutputField(
        desc=(
            'JSON array of 2-3 tool calls, e.g.: '
            '[{"tool": "topic_search", "args": {"topic": "Deathly Hallows", "query": "Elder Wand masters", "k": 5}}, '
            '{"tool": "character_search", "args": {"character": "albus-dumbledore", "query": "Elder Wand", "k": 4}}]'
        )
    )


class PerspectiveShiftSignature(dspy.Signature):
    """Extract a principle, philosophy, or behavioral pattern from a Harry Potter
    character based on the retrieved lore, then apply it to a real-world situation
    the user describes. Ground the character's traits in corpus facts, then reason
    about how those traits translate to practical advice or insight. Be specific
    and actionable — not generic motivational advice.

    Multi-turn aware: ``chat_history`` carries prior turns in this session where
    the user explored the same character's perspective on evolving facets of
    the scenario. When present, treat the current scenario as a continuation —
    build on the prior principle rather than reintroducing the character.
    """

    scenario:  str = dspy.InputField(desc="the real-world situation or question")
    character: str = dspy.InputField(desc="the HP character whose perspective to apply")
    context:   str = dspy.InputField(desc="retrieved lore passages about this character, each prefixed [doc_id]")
    chat_history: str = dspy.InputField(desc=(
        "Prior perspective-shift turns in this conversation (may be empty). "
        "When present, this is a continuation of an ongoing dialogue. "
        "Rules: NEVER open with a greeting ('Oh hello', 'Hello there', "
        "'Hi', 'Welcome'). NEVER reintroduce the character ('As Luna, I...'). "
        "NEVER reset the conversation — do not treat a terse reply ('nothing', "
        "'love', 'I don't know') as a new topic; interpret it as an answer to "
        "the prior turn's follow-up question. Directly engage with what the "
        "user just said, reference the prior turn if relevant, and continue "
        "the same unfolding conversation."
    ))

    character_principle: str = dspy.OutputField(
        desc=("2-3 sentences. The core principle this character embodies, anchored "
              "in a specific canon event or decision. Do NOT list traits or give a "
              "summary of the character — name the one lesson their life teaches.")
    )
    applied_insight: str = dspy.OutputField(
        desc=("3-4 sentences (roughly 60-90 words). A direct, actionable insight "
              "for the user's scenario. Name a concrete action or stance they can "
              "take this week. Avoid hedging ('it's important to...', 'you might "
              "consider...') and avoid restating the scenario.")
    )
    reasoning: str = dspy.OutputField(
        desc=("1-2 sentences. The bridge: why THIS character's specific experience "
              "maps to THIS scenario. Cite the connecting event, not a trait.")
    )
    character_response: str = dspy.OutputField(
        desc=("The character speaking DIRECTLY to the user, in first person, as "
              "if they are in the same room. 120-180 words. Synthesize the "
              "principle, insight, and reasoning above into the character's own "
              "voice — do NOT label sections, do NOT say 'my principle is...' or "
              "'applied to your situation...'. Weave canon events in naturally, "
              "the way a mentor would reference a past experience. Use 'I' and "
              "'you'. Match this character's known cadence and vocabulary "
              "(Dumbledore: reflective, aphoristic; McGonagall: direct, dry; "
              "Luna: oblique, unexpected). End with a single concrete question "
              "or stance — not a summary. "
              "If chat_history above is non-empty, this is a follow-up turn: "
              "do NOT open with a greeting, do NOT re-introduce yourself, do "
              "NOT reset to a cold-start tone. Directly engage with what the "
              "user said; if they replied tersely, interpret their reply as an "
              "answer to your prior question and continue the thread. "
              "When the user explicitly asks you to summarize, reflect back, "
              "paraphrase what they've shared, or recall their situation, open "
              "the response by restating their scenario in your own voice, still "
              "fully in character. Then transition to a brief insight or question. "
              "The reflection replaces the usual advice-first pattern for that "
              "turn only.")
    )
    citations: str = dspy.OutputField(
        desc=("Space-separated doc_ids formatted as [doc-id], each in square "
              "brackets. Use ONLY doc_ids that appear verbatim in the context "
              "field. Do not invent or partially-spell doc_ids. Example: "
              "'[severus-snape/biography-early-life-001] [severus-snape/personality-and-traits-002]'")
    )


class OpenAnalysisSignature(dspy.Signature):
    """Analytical mode: use retrieved lore as a factual foundation, then draw on
    your broader knowledge (psychology, literary theory, history, philosophy) to
    provide deep analysis. Clearly mark which parts come from the corpus vs your
    own reasoning. Do not refuse to answer — if the corpus is thin, lean on your
    general knowledge and say so.

    Multi-turn aware: ``chat_history`` carries prior exchanges in this session.
    When the user asks a follow-up like "so what did you mean earlier about X",
    use the history to pick up the thread rather than restarting from scratch.
    """

    question: str = dspy.InputField(desc="the analytical question")
    context:  str = dspy.InputField(desc="retrieved lore passages for grounding, each prefixed [doc_id]")
    chat_history: str = dspy.InputField(desc="prior analysis turns in this conversation (may be empty for a new thread). When present, treat the current question as a continuation — reference or refine your earlier points rather than repeating them.")

    analysis:       str = dspy.OutputField(desc="3-5 sentences. Direct analysis blending corpus facts with broader knowledge. Commit to a position. Avoid hedging ('it's important to...') and avoid restating the question.")
    corpus_facts:   str = dspy.OutputField(desc="2-3 sentences. ONLY facts drawn from [doc_id] chunks in context. Cite them.")
    own_reasoning:  str = dspy.OutputField(desc="2-3 sentences. Claims NOT in the corpus — your broader knowledge, explicitly marked as interpretation.")
    citations:      str = dspy.OutputField(desc="Space-separated doc_ids formatted as [doc-id], each in square brackets. Use ONLY doc_ids that appear in the context field. Output 'none' if the answer relies entirely on general knowledge.")


class ExamGraderSignature(dspy.Signature):
    """Grade a student's answer strictly against the retrieved textbook data.
    Deduct points for claims not supported by the context. Be specific in critique."""

    question:       str = dspy.InputField(desc="the exam question")
    student_answer: str = dspy.InputField(desc="the student's submitted answer")
    context:        str = dspy.InputField(desc="authoritative textbook passages, each prefixed [doc_id]")

    score:      int  = dspy.OutputField(desc="0-100, strictly based on context accuracy")
    is_passing: bool = dspy.OutputField(desc="true if score >= 60")
    critique:   str  = dspy.OutputField(desc="specific errors or omissions in the student's answer")
    citations:  str  = dspy.OutputField(desc="space-separated doc_ids used for grading")


class GuidedLearningSignature(dspy.Signature):
    """Socratic tutor. Guide the learner without revealing the answer directly.
    Use their past attempts and the prior conversation to personalise the hint.

    Two memory fields:
      - ``past_attempts``: the learner's own prior work on this problem,
        client-supplied (they may edit / re-submit their attempt).
      - ``chat_history``: the prior tutor-student exchanges in this session,
        server-populated from the conversation store. Use it to avoid
        repeating yourself and to pick up where you left off.
    """

    question:      str = dspy.InputField(desc="the learner's question")
    context:       str = dspy.InputField(desc="retrieved lore passages, each prefixed [doc_id]")
    past_attempts: str = dspy.InputField(desc="learner's prior answers/attempts on this specific problem, or 'none'")
    chat_history:  str = dspy.InputField(desc="previous tutor-student exchanges in this session (may be empty for a new conversation)")

    hint:          str = dspy.OutputField(desc="a guiding hint that does not give the answer away")
    next_question: str = dspy.OutputField(desc="a probing follow-up question ending with '?'")
    explanation:   str = dspy.OutputField(desc="concept explanation (2-4 sentences) without revealing the direct answer")


class DebateSignature(dspy.Signature):
    """Present both sides of a lore debate with canon evidence.
    Argue for and against the position, then deliver a verdict on which side
    the canon supports more strongly."""

    position: str = dspy.InputField(desc="the debatable claim or position about HP lore")
    context:  str = dspy.InputField(desc="retrieved lore passages, each prefixed [doc_id]")

    arguments_for:     str = dspy.OutputField(desc="canon-supported arguments in favour of the position")
    arguments_against: str = dspy.OutputField(desc="canon-supported arguments against the position")
    verdict:           str = dspy.OutputField(desc="which side the canon evidence supports more strongly, and why")
    citations:         str = dspy.OutputField(desc="space-separated doc_ids used across both sides of the argument")


class SatiricalPodcastSignature(dspy.Signature):
    """Generate a short satirical podcast transcript where two opinionated hosts
    discuss a Harry Potter topic through a modern, mundane lens. Ground magical
    elements in real canon facts from context, then subvert them with contemporary
    absurdity — consumerism, influencer culture, gig economy, dating apps, social
    status anxiety. The comedy lives in the collision between the fantastical and
    the embarrassingly relatable. Dark humour is welcome; stay grounded in canon."""

    topic:        str = dspy.InputField(desc="the HP magical topic or scenario being satirised")
    modern_angle: str = dspy.InputField(desc="the mundane contemporary lens, e.g. 'online marketplace', 'dating app ethics', 'gig economy labour rights'")
    context:      str = dspy.InputField(desc="retrieved lore passages, each prefixed [doc_id]")

    transcript:      str = dspy.OutputField(desc="podcast dialogue of 4-8 exchanges between two named hosts — each line formatted as 'Name: dialogue'. Must weave in specific canon facts and apply the modern_angle for comedic effect.")
    comedic_tension: str = dspy.OutputField(desc="one sentence: the central absurdity being exploited — what makes this magical element ridiculous through the modern lens")
    citations:       str = dspy.OutputField(desc="space-separated doc_ids used for canon grounding")


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class DeepResearchModule(dspy.Module):
    """
    Broad retrieval (k=10) + ChainOfThought over DeepResearchSignature.
    Optimizer metric: citation accuracy + source coverage.
    """

    def __init__(self, pipeline: RAGPipeline, k: int = 10) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._k = k
        self.predict = dspy.ChainOfThought(DeepResearchSignature)

    def forward(self, question: str, user_profile: Optional[Dict[str, Any]] = None) -> dspy.Prediction:
        chunks = self._pipeline.retrieve(question, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(question=question, context=context)


_MAX_MERGED_CHUNKS = 15   # cap total chunks fed to the synthesiser


def _parse_tool_calls(raw: str) -> list:
    """
    Extract a JSON array of tool calls from the LLM output.
    Handles three common failure modes:
      1. Clean JSON array — parsed directly.
      2. JSON wrapped in ```json ... ``` fences — fences stripped first.
      3. Unparseable string — returns empty list (caller falls back to vector_search).
    """
    import json, re
    text = raw.strip()
    # Strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    # Find the first [...] array in the string
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass
    return []


def _dedup_chunks(chunk_lists: list) -> list:
    """Merge multiple lists of chunk dicts, deduplicating by (doc_id, text[:60])."""
    seen: set = set()
    merged: list = []
    for chunks in chunk_lists:
        for c in chunks:
            key = (c["doc_id"], c["text"][:60])
            if key not in seen:
                seen.add(key)
                merged.append(c)
    return merged[:_MAX_MERGED_CHUNKS]


class MultiToolDeepResearchModule(dspy.Module):
    """
    Two-step deep research:
      1. QueryPlanSignature — LLM plans 2-3 targeted tool calls.
      2. Execute plan via ToolRegistry (validated, logged).
      3. DeepResearchSignature — synthesise merged context into answer.

    Falls back to single vector_search if the plan cannot be parsed.
    """

    def __init__(self, pipeline: RAGPipeline, char_pipeline=None) -> None:
        super().__init__()
        self._registry = build_retrieval_registry(pipeline, char_pipeline=char_pipeline)
        self._plan = dspy.ChainOfThought(QueryPlanSignature)
        self._synthesise = dspy.ChainOfThought(DeepResearchSignature)

    def forward(self, question: str, **kwargs) -> dspy.Prediction:
        # Step 1: plan
        plan_pred = self._plan(question=question)
        tool_calls = _parse_tool_calls(plan_pred.tool_calls)

        # Step 2: execute; fall back to vector_search if plan is empty/invalid
        if not tool_calls:
            tool_calls = [{"tool": "vector_search", "args": {"query": question, "k": 10}}]

        chunk_lists = []
        for call in tool_calls[:3]:  # max 3 calls
            name = call.get("tool", "vector_search")
            args = call.get("args", {})
            # Ensure k is an int and within bounds
            if "k" in args:
                args["k"] = max(1, min(int(args["k"]), 15))
            try:
                result = self._registry.call_sync(name, args)
                chunk_lists.append(result)
            except Exception:
                # Bad tool call — skip, don't crash the whole request
                pass

        if not chunk_lists:
            # All calls failed — last-resort fallback
            fallback = self._registry.call_sync("vector_search", {"query": question, "k": 10})
            chunk_lists.append(fallback)

        # Step 3: synthesise
        chunks = _dedup_chunks(chunk_lists)
        context = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in chunks) or "No context retrieved."
        return self._synthesise(question=question, context=context)


_CHARACTER_SLUG: Dict[str, str] = {
    "harry":        "harry-potter",
    "harry potter": "harry-potter",
    "hermione":     "hermione-granger",
    "hermione granger": "hermione-granger",
    "ron":          "ron-weasley",
    "ron weasley":  "ron-weasley",
    "dumbledore":   "albus-dumbledore",
    "albus":        "albus-dumbledore",
    "albus dumbledore": "albus-dumbledore",
    "snape":        "severus-snape",
    "severus":      "severus-snape",
    "severus snape": "severus-snape",
    "mcgonagall":   "minerva-mcgonagall",
    "minerva":      "minerva-mcgonagall",
    "minerva mcgonagall": "minerva-mcgonagall",
    "luna":         "luna-lovegood",
    "luna lovegood": "luna-lovegood",
    "neville":      "neville-longbottom",
    "neville longbottom": "neville-longbottom",
    "voldemort":    "lord-voldemort",
    "lord voldemort": "lord-voldemort",
    "tom riddle":   "lord-voldemort",
    "draco":        "draco-malfoy",
    "draco malfoy": "draco-malfoy",
    "hagrid":       "rubeus-hagrid",
    "rubeus hagrid": "rubeus-hagrid",
}


def _normalize_character(name: str) -> str:
    """Convert a free-form character name ('Dumbledore', 'Harry Potter') into
    the canonical lowercase-hyphenated slug used in the character_lore
    collection's metadata ('albus-dumbledore', 'harry-potter')."""
    key = (name or "").strip().lower()
    return _CHARACTER_SLUG.get(key, key.replace(" ", "-"))


import re as _re

def _detect_character_in_question(question: str) -> Optional[str]:
    """Scan a free-form question for any known character alias.

    Returns the canonical slug of the first match (longest-alias-first
    to avoid "Albus" winning over "Albus Dumbledore"), or ``None`` if
    no character is mentioned.

    Uses word-boundary matching so "Ron" doesn't match "Ronald"-style
    substrings.
    """
    if not question:
        return None
    q_lower = question.lower()
    # Match longer aliases first ("albus dumbledore" beats "albus")
    for alias in sorted(_CHARACTER_SLUG.keys(), key=len, reverse=True):
        if _re.search(rf"\b{_re.escape(alias)}\b", q_lower):
            return _CHARACTER_SLUG[alias]
    return None


class PerspectiveShiftModule(dspy.Module):
    """
    Retrieves lore about a specific character, then applies their philosophy
    to a real-world scenario via ChainOfThought.

    Retrieval strategy
    ------------------
    If ``char_pipeline`` is provided, use it with a
    ``where={"character": <slug>}`` filter — chunks in the ``character_lore``
    collection carry that metadata, so retrieval is hard-scoped to the
    requested character. Fallback to the generic ``pipeline`` only when the
    filtered retrieval returns nothing (covers characters not present in
    the character corpus).
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        k: int = 5,
        char_pipeline: Optional[RAGPipeline] = None,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._char_pipeline = char_pipeline
        self._k = k
        self.predict = dspy.ChainOfThought(PerspectiveShiftSignature)

    def forward(
        self,
        scenario: str = "",
        character: str = "Dumbledore",
        chat_history: str = "",
        **kwargs,
    ) -> dspy.Prediction:
        slug = _normalize_character(character)
        chunks: list = []

        if self._char_pipeline is not None:
            chunks = self._char_pipeline.retrieve(
                scenario,
                top_k=self._k,
                where={"character": slug},
            )
        if not chunks:
            # Fallback: generic retrieval on the default corpus
            query = f"{character} {scenario}".strip()
            chunks = self._pipeline.retrieve(query, top_k=self._k)

        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(
            character=character,
            scenario=scenario,
            context=context,
            chat_history=chat_history,
        )


class OpenAnalysisModule(dspy.Module):
    """
    Broad retrieval (k=7) + ChainOfThought over OpenAnalysisSignature.

    Character-aware retrieval: when the question mentions a known
    character AND a ``char_pipeline`` (character_lore collection) is
    available, retrieval splits 5:2 between that character's chunks and
    the general corpus. This addresses the observed grounding weakness
    where analysis questions about a specific character ("why did Snape
    become Snape") pulled from a generic 10-doc corpus with ~2-3 shallow
    facts per chunk.

    Falls back to pure default-corpus retrieval (k=7) when no character
    is detected or no ``char_pipeline`` was wired in.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        k: int = 7,
        char_pipeline: Optional[RAGPipeline] = None,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._char_pipeline = char_pipeline
        self._k = k
        self.predict = dspy.ChainOfThought(OpenAnalysisSignature)

    def forward(self, question: str, chat_history: str = "", **kwargs) -> dspy.Prediction:
        character_slug = _detect_character_in_question(question)

        if character_slug and self._char_pipeline is not None:
            # Split retrieval: most slots to character-specific chunks, a few
            # for general-corpus context (institutions, events, themes that
            # aren't character-scoped).
            char_chunks = self._char_pipeline.retrieve(
                question, top_k=5, where={"character": character_slug}
            )
            general_chunks = self._pipeline.retrieve(question, top_k=2)
            chunks = list(char_chunks) + list(general_chunks)
        else:
            chunks = self._pipeline.retrieve(question, top_k=self._k)

        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(question=question, context=context, chat_history=chat_history)


class ExamGraderModule(dspy.Module):
    """
    Strict grading: retrieve authoritative chunks (k=5), grade the student's
    answer against them. The optimizer metric penalises lenient grading of
    wrong answers and harsh grading of correct ones.
    """

    def __init__(self, pipeline: RAGPipeline, k: int = 5) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._k = k
        self.predict = dspy.ChainOfThought(ExamGraderSignature)

    def forward(self, question: str, student_answer: str = "", **kwargs) -> dspy.Prediction:
        chunks = self._pipeline.retrieve(question, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(question=question, student_answer=student_answer, context=context)


class GuidedLearningModule(dspy.Module):
    """
    Narrow retrieval (k=3) biased toward past-attempt context
    + ChainOfThought over GuidedLearningSignature.
    Optimizer metric: Socratic score (penalises giving the answer directly).
    """

    def __init__(self, pipeline: RAGPipeline, k: int = 3) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._k = k
        self.predict = dspy.ChainOfThought(GuidedLearningSignature)

    def forward(
        self,
        question: str,
        concept: str = "",
        past_attempts: str = "none",
        chat_history: str = "",
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> dspy.Prediction:
        query = f"{concept} {question}".strip()
        chunks = self._pipeline.retrieve(query, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(
            question=question,
            context=context,
            past_attempts=past_attempts,
            chat_history=chat_history,
        )


class DebateModule(dspy.Module):
    """
    Balanced retrieval (k=7) + ChainOfThought over DebateSignature.
    Optimizer metric: both sides must cite distinct canon passages; verdict
    must not be empty and must reference at least one doc_id.
    """

    def __init__(self, pipeline: RAGPipeline, k: int = 7) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._k = k
        self.predict = dspy.ChainOfThought(DebateSignature)

    def forward(self, position: str = "", **kwargs) -> dspy.Prediction:
        chunks = self._pipeline.retrieve(position, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(position=position, context=context)


class SatiricalPodcastModule(dspy.Module):
    """
    Moderate retrieval (k=6) + ChainOfThought over SatiricalPodcastSignature.
    Retrieves on the topic to ground the script in real canon, then the LLM
    applies the modern_angle for satirical effect.
    Optimizer metric: dialogue structure + substance + canon citation present.
    """

    def __init__(self, pipeline: RAGPipeline, k: int = 6) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._k = k
        self.predict = dspy.ChainOfThought(SatiricalPodcastSignature)

    def forward(
        self,
        topic: str = "",
        modern_angle: str = "modern life",
        **kwargs,
    ) -> dspy.Prediction:
        chunks = self._pipeline.retrieve(topic, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(topic=topic, modern_angle=modern_angle, context=context)


# ---------------------------------------------------------------------------
# Agent — mode router
# ---------------------------------------------------------------------------

MODES = {"deep_research", "guided_learning", "exam_grader", "open_analysis",
         "perspective_shift", "debate", "satirical_podcast"}


class DSPyAgent:
    """
    Routes queries to the correct compiled DSPy module.

    Each mode is a separate Module + compiled JSON artifact. Both are needed
    for full portability: ChromaDB snapshot + compiled JSONs = your .agent export.

    Usage:
        agent = DSPyAgent(pipeline)
        result = agent.forward("deep_research", "Who are the Peverell brothers?")
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        export_dir: Optional[str] = None,
        research_k: int = 10,
        learning_k: int = 3,
        char_pipeline: Optional[RAGPipeline] = None,
    ) -> None:
        self._pipeline = pipeline
        # Best-effort: if no char_pipeline was passed explicitly, try to
        # attach to a `character_lore` collection on the same Chroma client.
        # Silently no-op if the collection doesn't exist (e.g. fresh install
        # that hasn't run ingest_character_lore yet).
        if char_pipeline is None:
            char_pipeline = _try_build_character_pipeline(pipeline)
        self._char_pipeline = char_pipeline

        self._modules: Dict[str, dspy.Module] = {
            "deep_research":   MultiToolDeepResearchModule(pipeline, char_pipeline=char_pipeline),
            "guided_learning": GuidedLearningModule(pipeline, k=learning_k),
            "exam_grader":     ExamGraderModule(pipeline, k=5),
            "open_analysis":   OpenAnalysisModule(
                pipeline, k=7, char_pipeline=char_pipeline,
            ),
            "perspective_shift": PerspectiveShiftModule(
                pipeline, k=5, char_pipeline=char_pipeline,
            ),
            "debate":            DebateModule(pipeline, k=7),
            "satirical_podcast": SatiricalPodcastModule(pipeline, k=6),
        }
        self._router = IntentRouterModule()
        if export_dir:
            self._load(Path(export_dir))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # Capability list returned when the router classifies a message as "none"
    _CAPABILITY_TEXT = (
        "I'm a Harry Potter lore agent! Here's what I can do:\n\n"
        "- **Ask a question** — factual HP lore lookup (deep_research)\n"
        "- **Learn interactively** — Socratic tutoring on HP topics (guided_learning)\n"
        "- **Get graded** — submit an answer for canon-based grading (exam_grader)\n"
        "- **Analyze** — deep analysis blending canon with broader knowledge (open_analysis)\n"
        "- **Character advice** — get advice from an HP character's perspective (perspective_shift)\n"
        "- **Debate** — argue both sides of a HP lore debate (debate)\n"
        "- **Comedy** — satirical podcast on HP topics through a modern lens (satirical_podcast)\n\n"
        "Try asking something about the Harry Potter universe!"
    )

    def forward(self, mode: str, text: str = "", **kwargs) -> dspy.Prediction:
        """
        Route a query to the correct mode.

        When ``mode="auto"``, the intent router classifies the user's message
        and dispatches to the resolved mode. The returned prediction carries
        ``routed_mode`` and ``router_confidence`` attributes.

        `text` maps to the Signature's **primary** input field (the first
        non-`context` input) via introspection. Secondary inputs — e.g.
        `character`, `student_answer`, `past_attempts`, `modern_angle` — must
        be passed as explicit kwargs.
        """
        if mode == "auto":
            return self._route_auto(text, **kwargs)

        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}. Valid modes: {sorted(MODES)}")
        module = self._modules[mode]
        primary = _primary_input_field(module)
        if text and primary not in kwargs:
            kwargs[primary] = text
        return module.forward(**kwargs)

    def route_only(self, text: str) -> Dict[str, Any]:
        """Run the intent router *without* dispatching to a mode module.

        Returns ``{"mode": str, "confidence": str, "kwargs": dict}`` — the
        same shape ``_route_auto`` consumes internally. Used by the API layer
        to inject stickiness/policy between routing and dispatch (see
        SPEC.md §Design): caller decides the effective mode, then calls
        ``forward(effective_mode, ...)`` to dispatch.
        """
        router_pred = self._router.forward(user_message=text)
        return parse_router_output(router_pred)

    def _route_auto(self, text: str, **kwargs) -> dspy.Prediction:
        """Classify intent via the router, then dispatch to the resolved mode."""
        router_pred = self._router.forward(user_message=text)
        result = parse_router_output(router_pred)

        if result["mode"] == "none":
            pred = dspy.Prediction(answer=self._CAPABILITY_TEXT)
            pred.routed_mode = "none"
            pred.router_confidence = result["confidence"]
            return pred

        # Merge: router-extracted kwargs are defaults; caller kwargs override
        merged_kwargs = {**result["kwargs"], **kwargs}
        resolved_mode = result["mode"]

        pred = self.forward(resolved_mode, text, **merged_kwargs)
        pred.routed_mode = resolved_mode
        pred.router_confidence = result["confidence"]
        return pred

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def save(self, export_dir: str) -> None:
        """Persist compiled programs and manifest to <export_dir>/."""
        path = Path(export_dir)
        path.mkdir(parents=True, exist_ok=True)

        for mode, module in self._modules.items():
            module.save(str(path / f"{mode}.json"))

        self._router.save(str(path / "intent_router.json"))

        _write_manifest(path, embedding_model=self._pipeline.embedding_model_name)

    def _load(self, path: Path) -> None:
        """Load compiled programs after validating manifest."""
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            _validate_manifest(manifest, self._pipeline.embedding_model_name)

        for mode, module in self._modules.items():
            artifact = path / f"{mode}.json"
            if artifact.exists():
                module.load(str(artifact))

        router_artifact = path / "intent_router.json"
        if router_artifact.exists():
            self._router.load(str(router_artifact))

    # ------------------------------------------------------------------
    # Optimiser helpers (called during training, not inference)
    # ------------------------------------------------------------------

    def compile_deep_research(self, optimizer: dspy.teleprompt.Teleprompter, trainset: list) -> None:
        """Run the optimizer on the deep_research (multi-tool) module and update in place."""
        self._modules["deep_research"] = optimizer.compile(
            self._modules["deep_research"], trainset=trainset
        )

    def compile_guided_learning(self, optimizer: dspy.teleprompt.Teleprompter, trainset: list) -> None:
        """Run the optimizer on the guided_learning module and update in place."""
        self._modules["guided_learning"] = optimizer.compile(
            self._modules["guided_learning"], trainset=trainset
        )

    def compile_exam_grader(self, optimizer: dspy.teleprompt.Teleprompter, trainset: list) -> None:
        """Run the optimizer on the exam_grader module and update in place."""
        self._modules["exam_grader"] = optimizer.compile(
            self._modules["exam_grader"], trainset=trainset
        )

    def compile_debate(self, optimizer: dspy.teleprompt.Teleprompter, trainset: list) -> None:
        """Run the optimizer on the debate module and update in place."""
        self._modules["debate"] = optimizer.compile(
            self._modules["debate"], trainset=trainset
        )

    def compile_satirical_podcast(self, optimizer: dspy.teleprompt.Teleprompter, trainset: list) -> None:
        """Run the optimizer on the satirical_podcast module and update in place."""
        self._modules["satirical_podcast"] = optimizer.compile(
            self._modules["satirical_podcast"], trainset=trainset
        )

    def compile_perspective_shift(self, optimizer: dspy.teleprompt.Teleprompter, trainset: list) -> None:
        """Run the optimizer on the perspective_shift module and update in place.

        Uses both the single-turn and multi-turn examples in
        data/trainset_perspective_shift.py. The multi-turn examples are
        the ones that teach "no greeting on turn 2+" — the metric
        (perspective_shift_metric) enforces this at compile time.
        """
        self._modules["perspective_shift"] = optimizer.compile(
            self._modules["perspective_shift"], trainset=trainset
        )

    def compile_intent_router(self, optimizer: dspy.teleprompt.Teleprompter, trainset: list) -> None:
        """Run the optimizer on the intent router module and update in place."""
        self._router = optimizer.compile(
            self._router, trainset=trainset
        )


# ---------------------------------------------------------------------------
# Router helpers
# ---------------------------------------------------------------------------

def _try_build_character_pipeline(default_pipeline: RAGPipeline) -> Optional[RAGPipeline]:
    """Attach a second RAGPipeline to the ``character_lore`` Chroma collection.

    Returns ``None`` silently if the collection doesn't exist (e.g. a fresh
    install where ``scripts/ingest_character_lore.py`` hasn't been run).
    Shares the same Chroma client + embedding function as the default
    pipeline — no extra connections, no new config.
    """
    try:
        client = getattr(default_pipeline, "_client", None)
        ef = getattr(default_pipeline, "_ef", None)
        if client is None or ef is None:
            return None
        # Check the collection exists before building a second pipeline
        existing = {c.name for c in client.list_collections()}
        if "character_lore" not in existing:
            return None
        from .ingest_lore import build_pipeline as _build
        # build_pipeline will point at the character_lore collection
        char_pipeline = _build(persist=True, collection_name="character_lore")
        return char_pipeline
    except Exception:
        return None


def _primary_input_field(module: dspy.Module) -> str:
    """
    Return the first non-`context` input field name from the module's Signature.

    Single-predictor modules (all standard modes) expose a `predict` attribute
    (a ChainOfThought). Multi-predictor modules (MultiToolDeepResearchModule)
    don't have a single `predict` — fall back to the first Predict found via
    module.predictors(), which returns all inner Predict instances in order.
    """
    if hasattr(module, 'predict'):
        sig = module.predict.predictors()[0].signature
    else:
        preds = module.predictors()
        if not preds:
            raise RuntimeError(
                f"Signature for {type(module).__name__} has no predictors"
            )
        sig = preds[0].signature
    for name in sig.input_fields:
        if name != "context":
            return name
    raise RuntimeError(
        f"Signature for {type(module).__name__} has no primary input field"
    )


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _write_manifest(path: Path, embedding_model: str = _DEFAULT_EMBEDDING_MODEL) -> None:
    hashes = {}
    for mode in MODES:
        artifact = path / f"{mode}.json"
        if artifact.exists():
            hashes[mode] = _file_sha256(artifact)

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "embedding_model": embedding_model,
        "program_hashes": hashes,
        "created_at": time.time(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _validate_manifest(manifest: dict, current_embedding_model: str) -> None:
    """Raise BackfillRequired if the manifest is stale."""
    stored_model = manifest.get("embedding_model", "")
    if stored_model and stored_model != current_embedding_model:
        raise BackfillRequired(
            f"Embedding model changed: manifest has {stored_model!r}, "
            f"runtime uses {current_embedding_model!r}. Re-embed before loading."
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BackfillRequired(RuntimeError):
    """Raised when manifest detects an embedding model mismatch."""


# ---------------------------------------------------------------------------
# RAGPipeline extension — expose embedding model name
# ---------------------------------------------------------------------------
# Monkey-patch a property onto RAGPipeline so DSPyAgent can read the model name
# without importing it from a different layer.

if not hasattr(RAGPipeline, "embedding_model_name"):
    RAGPipeline.embedding_model_name = property(  # type: ignore[assignment]
        lambda self: getattr(self, "_embedding_model_name", _DEFAULT_EMBEDDING_MODEL)
    )
    _orig_init = RAGPipeline.__init__

    def _patched_init(self, *args, **kwargs):
        self._embedding_model_name = kwargs.get("embedding_model", _DEFAULT_EMBEDDING_MODEL)
        _orig_init(self, *args, **kwargs)

    RAGPipeline.__init__ = _patched_init  # type: ignore[assignment]

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

from .rag_pipeline import RAGPipeline


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


class PerspectiveShiftSignature(dspy.Signature):
    """Extract a principle, philosophy, or behavioral pattern from a Harry Potter
    character based on the retrieved lore, then apply it to a real-world situation
    the user describes. Ground the character's traits in corpus facts, then reason
    about how those traits translate to practical advice or insight. Be specific
    and actionable — not generic motivational advice."""

    character: str = dspy.InputField(desc="the HP character whose perspective to apply")
    scenario:  str = dspy.InputField(desc="the real-world situation or question")
    context:   str = dspy.InputField(desc="retrieved lore about this character, each prefixed [doc_id]")

    character_principle: str = dspy.OutputField(desc="the core principle or philosophy this character embodies, grounded in specific canon events")
    applied_insight:     str = dspy.OutputField(desc="how this principle applies to the user's real-world scenario — specific, actionable, not generic")
    reasoning:           str = dspy.OutputField(desc="the bridge: why this character's experience maps to this situation")
    citations:           str = dspy.OutputField(desc="space-separated doc_ids used for character grounding")


class OpenAnalysisSignature(dspy.Signature):
    """Analytical mode: use retrieved lore as a factual foundation, then draw on
    your broader knowledge (psychology, literary theory, history, philosophy) to
    provide deep analysis. Clearly mark which parts come from the corpus vs your
    own reasoning. Do not refuse to answer — if the corpus is thin, lean on your
    general knowledge and say so."""

    question: str = dspy.InputField(desc="the analytical question")
    context:  str = dspy.InputField(desc="retrieved lore passages for grounding, each prefixed [doc_id]")

    analysis:       str = dspy.OutputField(desc="in-depth analysis blending corpus facts with broader knowledge")
    corpus_facts:   str = dspy.OutputField(desc="key facts drawn from the retrieved context")
    own_reasoning:  str = dspy.OutputField(desc="interpretations, theories, or analysis beyond the corpus")
    citations:      str = dspy.OutputField(desc="space-separated doc_ids referenced, or 'none' if mostly general knowledge")


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
    Use their past attempts to personalise the hint."""

    question:     str = dspy.InputField(desc="the learner's question")
    context:      str = dspy.InputField(desc="relevant lore context for this concept")
    past_attempts: str = dspy.InputField(desc="learner's prior answers/attempts, or 'none'")

    hint:          str = dspy.OutputField(desc="a guiding hint that does not give the answer away")
    next_question: str = dspy.OutputField(desc="a follow-up question to deepen understanding")
    explanation:   str = dspy.OutputField(desc="concept explanation without revealing the direct answer")


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


class PerspectiveShiftModule(dspy.Module):
    """
    Retrieves lore about a specific character (k=5), then applies their
    philosophy to a real-world scenario via ChainOfThought.
    """

    def __init__(self, pipeline: RAGPipeline, k: int = 5) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._k = k
        self.predict = dspy.ChainOfThought(PerspectiveShiftSignature)

    def forward(self, question: str, character: str = "Dumbledore", **kwargs) -> dspy.Prediction:
        query = f"{character} {question}".strip()
        chunks = self._pipeline.retrieve(query, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(character=character, scenario=question, context=context)


class OpenAnalysisModule(dspy.Module):
    """
    Broad retrieval (k=7) + ChainOfThought over OpenAnalysisSignature.
    Uses corpus as grounding but allows the LLM to reason beyond it.
    """

    def __init__(self, pipeline: RAGPipeline, k: int = 7) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._k = k
        self.predict = dspy.ChainOfThought(OpenAnalysisSignature)

    def forward(self, question: str, **kwargs) -> dspy.Prediction:
        chunks = self._pipeline.retrieve(question, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(question=question, context=context)


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
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> dspy.Prediction:
        query = f"{concept} {question}".strip()
        chunks = self._pipeline.retrieve(query, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(question=question, context=context, past_attempts=past_attempts)


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

    def forward(self, question: str, user_profile: Optional[Dict[str, Any]] = None) -> dspy.Prediction:
        chunks = self._pipeline.retrieve(question, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(position=question, context=context)


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
        question: str,
        modern_angle: str = "modern life",
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> dspy.Prediction:
        chunks = self._pipeline.retrieve(question, top_k=self._k)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks) or "No context retrieved."
        return self.predict(topic=question, modern_angle=modern_angle, context=context)


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
    ) -> None:
        self._pipeline = pipeline
        self._modules: Dict[str, dspy.Module] = {
            "deep_research":   DeepResearchModule(pipeline, k=research_k),
            "guided_learning": GuidedLearningModule(pipeline, k=learning_k),
            "exam_grader":     ExamGraderModule(pipeline, k=5),
            "open_analysis":   OpenAnalysisModule(pipeline, k=7),
            "perspective_shift": PerspectiveShiftModule(pipeline, k=5),
            "debate":            DebateModule(pipeline, k=7),
            "satirical_podcast": SatiricalPodcastModule(pipeline, k=6),
        }
        if export_dir:
            self._load(Path(export_dir))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, mode: str, question: str, **kwargs) -> dspy.Prediction:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}. Valid modes: {sorted(MODES)}")
        return self._modules[mode].forward(question=question, **kwargs)

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def save(self, export_dir: str) -> None:
        """Persist compiled programs and manifest to <export_dir>/."""
        path = Path(export_dir)
        path.mkdir(parents=True, exist_ok=True)

        for mode, module in self._modules.items():
            module.save(str(path / f"{mode}.json"))

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

    # ------------------------------------------------------------------
    # Optimiser helpers (called during training, not inference)
    # ------------------------------------------------------------------

    def compile_deep_research(self, optimizer: dspy.teleprompt.Teleprompter, trainset: list) -> None:
        """Run the optimizer on the deep_research module and update in place."""
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

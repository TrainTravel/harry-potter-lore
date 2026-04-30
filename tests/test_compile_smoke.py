"""
Compile-path smoke tests
========================
Each mode must survive `BootstrapFewShot.compile(module, trainset=...)` under
DummyLM. This catches Signature-vs-Module.forward drift (e.g. trainset
declares `.with_inputs("position")` but `forward()` only accepts `question`)
which the normal runtime-path tests cannot catch.

Rule enforced: the Signature's input field names are canonical. Module.forward
parameter names and Trainset `.with_inputs(...)` must be identical strings.

Tests use DummyLM so they never call a real API. A 2-example trainset slice
keeps each test under a second.
"""

from __future__ import annotations

import pytest
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.utils import DummyLM

from context_harness.dspy_agent import (
    DeepResearchModule,
    GuidedLearningModule,
    ExamGraderModule,
    DebateModule,
    SatiricalPodcastModule,
    PerspectiveShiftModule,
    _primary_input_field,
)
from context_harness.intent_router import IntentRouterModule
from context_harness.metrics import (
    deep_research_metric,
    socratic_metric,
    exam_grader_metric,
    debate_metric,
    satirical_podcast_metric,
    perspective_shift_metric,
    intent_router_metric,
)
from context_harness.ingest_lore import build_pipeline, parse_lore_file, LORE_FILE


# ---------------------------------------------------------------------------
# DummyLM answers — must cover every output field of every Signature we compile
# ---------------------------------------------------------------------------

_DUMMY_ANSWER = {
    # deep_research
    "answer": "Harry defeated Voldemort at the Battle of Hogwarts.",
    "citations": "harry-potter lord-voldemort",
    "confidence": "high",
    "gaps": "none",
    # guided_learning
    "hint": "Consider what happens when a wizard fears death above all else.",
    "next_question": "What class of dark magic splits a soul?",
    "explanation": "The concept touches on dark magic involving fragmentation of self.",
    # exam_grader
    "score": 75,
    "is_passing": True,
    "critique": "The answer covers the main idea but misses two supporting details from the text.",
    # debate
    "arguments_for": "The canon supports this claim across multiple events in the series.",
    "arguments_against": "However there are equally strong counter-examples throughout the books.",
    "verdict": "Canon evidence leans toward the claim, with caveats.",
    # satirical_podcast
    "transcript": (
        "Lavender: So I ordered the cloak on WizAmazon — Prime Owl delivery.\n"
        "Parvati: The owl retired halfway through. Classic.\n"
        "Lavender: And it arrived listed as used but visible. Fraud.\n"
        "Parvati: Leave a one-star review, they'll send a Howler."
    ),
    "comedic_tension": "An invisible product cannot be quality-checked, making reviews impossible.",
    # perspective_shift
    "character_principle": "Dumbledore consistently chose what was right over what was easy, even at great personal cost.",
    "applied_insight": "When paralyzed by a decision, identify the option whose difficulty comes from moral weight rather than mere inconvenience — that is often the right one.",
    "reasoning": "Dumbledore's choice to withhold the prophecy and bear its weight alone maps to any situation where integrity demands carrying a burden rather than offloading it.",
    "character_response": (
        "I know that weight well. The question is not whether you carry it — "
        "you will. The question is whether you carry it for the right reasons. "
        "Do one thing tomorrow. Not a grand thing. One small brave thing."
    ),
    # intent_router
    "mode": "deep_research",
    "kwargs_json": "{}",
    # reasoning trace (DSPy ChainOfThought always emits this)
    "reasoning": "Step-by-step reasoning trace.",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    """In-memory ChromaDB with all HP lore ingested once per module."""
    from context_harness.document_registry import DocumentRegistry
    p = build_pipeline(persist=False)
    reg = DocumentRegistry(p, db_path=":memory:")
    for doc_id, text in parse_lore_file(LORE_FILE):
        reg.upsert(doc_id, text)
    return p


@pytest.fixture(autouse=True)
def dummy_lm():
    """DummyLM configured with every output field every mode could ask for."""
    lm = DummyLM(answers=[_DUMMY_ANSWER])
    dspy.configure(lm=lm)
    yield lm


def _optimizer(metric, demos: int = 1):
    """Small optimizer: only bootstrap 1 demo, use 1 labelled demo."""
    return BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=demos,
        max_labeled_demos=demos,
    )


# ---------------------------------------------------------------------------
# Per-mode compile smoke tests
# ---------------------------------------------------------------------------

def test_deep_research_compile(pipeline):
    from data.trainset_deep_research import TRAINSET
    module = DeepResearchModule(pipeline, k=2)
    compiled = _optimizer(deep_research_metric).compile(module, trainset=TRAINSET[:2])
    assert compiled is not None


def test_guided_learning_compile(pipeline):
    from data.trainset_guided_learning import TRAINSET
    module = GuidedLearningModule(pipeline, k=2)
    compiled = _optimizer(socratic_metric).compile(module, trainset=TRAINSET[:2])
    assert compiled is not None


def test_exam_grader_compile(pipeline):
    from data.trainset_exam_grader import TRAINSET
    module = ExamGraderModule(pipeline, k=2)
    compiled = _optimizer(exam_grader_metric).compile(module, trainset=TRAINSET[:2])
    assert compiled is not None


def test_debate_compile(pipeline):
    """Regression test for the 2026-04-17 bug: DebateModule.forward must accept
    the Signature's input field name `position` — not `question`."""
    from data.trainset_debate import TRAINSET
    module = DebateModule(pipeline, k=2)
    compiled = _optimizer(debate_metric).compile(module, trainset=TRAINSET[:2])
    assert compiled is not None


def test_satirical_podcast_compile(pipeline):
    """Regression test: SatiricalPodcastModule.forward must accept `topic` (not
    `question`), matching the Signature's input field name."""
    from data.trainset_satirical_podcast import TRAINSET
    module = SatiricalPodcastModule(pipeline, k=2)
    compiled = _optimizer(satirical_podcast_metric).compile(module, trainset=TRAINSET[:2])
    assert compiled is not None


def test_perspective_shift_compile(pipeline):
    """Regression test: PerspectiveShiftModule.forward must accept `scenario`
    (the Signature's primary input field), and the trainset's `.with_inputs(
    "scenario", "character")` must match the module's kwargs."""
    from data.trainset_perspective_shift import TRAINSET
    module = PerspectiveShiftModule(pipeline, k=2)
    compiled = _optimizer(perspective_shift_metric).compile(module, trainset=TRAINSET[:2])
    assert compiled is not None


# ---------------------------------------------------------------------------
# Canonical-rule invariant: router introspection must find a primary input
# field for every mode, and that field must be a real forward() parameter.
# ---------------------------------------------------------------------------

def test_intent_router_compile():
    """IntentRouterModule must survive BootstrapFewShot under DummyLM."""
    from data.trainset_intent_router import TRAINSET
    module = IntentRouterModule()
    compiled = _optimizer(intent_router_metric).compile(module, trainset=TRAINSET[:2])
    assert compiled is not None


def test_intent_router_primary_input_field():
    """Verify Signature primary input is 'user_message' and forward() accepts it."""
    import inspect
    module = IntentRouterModule()
    sig = module.predict.predictors()[0].signature
    non_context_inputs = [name for name in sig.input_fields if name != "context"]
    assert non_context_inputs[0] == "user_message", (
        f"IntentRouterSignature primary input is {non_context_inputs[0]!r}, "
        f"expected 'user_message'"
    )
    params = inspect.signature(module.forward).parameters
    assert "user_message" in params, (
        "IntentRouterModule.forward() has no parameter named 'user_message'. "
        "This violates the Signature-is-canonical rule."
    )


@pytest.mark.parametrize("module_cls,expected_primary", [
    (DeepResearchModule,     "question"),
    (GuidedLearningModule,   "question"),
    (ExamGraderModule,       "question"),
    (DebateModule,           "position"),
    (SatiricalPodcastModule, "topic"),
    (PerspectiveShiftModule, "scenario"),
])
def test_primary_input_field_matches_forward_param(pipeline, module_cls, expected_primary):
    """The router's introspection must agree with the Module's forward() signature."""
    import inspect
    module = module_cls(pipeline, k=2)
    primary = _primary_input_field(module)
    assert primary == expected_primary, (
        f"{module_cls.__name__}: Signature's primary input is {primary!r} "
        f"but we expected {expected_primary!r}"
    )
    # The primary field name MUST appear as a parameter on forward() —
    # otherwise the router will crash when it tries to pass it.
    params = inspect.signature(module.forward).parameters
    assert primary in params, (
        f"{module_cls.__name__}.forward() has no parameter named {primary!r}. "
        f"This violates the Signature-is-canonical rule — the module's forward() "
        f"param names must match the Signature's input field names verbatim."
    )

"""Tests for data/trainset_*.py and context_harness/metrics.py"""

from types import SimpleNamespace
import pytest

from context_harness.metrics import (
    deep_research_metric, deep_research_strict_metric,
    socratic_metric, socratic_score,
    debate_metric, debate_score,
)


VALID_DOC_IDS = {
    "albus-dumbledore", "harry-potter", "hermione-granger", "ron-weasley",
    "lord-voldemort", "severus-snape", "hogwarts", "horcruxes",
    "deathly-hallows", "order-of-the-phoenix",
}


# ---------------------------------------------------------------------------
# Trainset integrity
# ---------------------------------------------------------------------------

def test_deep_research_trainset_nonempty():
    from data.trainset_deep_research import TRAINSET
    assert len(TRAINSET) >= 15


def test_deep_research_evalset_disjoint_from_train():
    from data.trainset_deep_research import TRAINSET, EVALSET
    train_qs = {ex.question for ex in TRAINSET}
    eval_qs  = {ex.question for ex in EVALSET}
    assert train_qs.isdisjoint(eval_qs)


def test_deep_research_citations_reference_valid_docs():
    from data.trainset_deep_research import TRAINSET, EVALSET
    for ex in TRAINSET + EVALSET:
        for cite in ex.citations.split():
            assert cite in VALID_DOC_IDS, f"unknown citation {cite!r} in question: {ex.question!r}"


def test_deep_research_examples_mark_question_as_input():
    from data.trainset_deep_research import TRAINSET
    for ex in TRAINSET:
        assert "question" in ex.inputs().keys()


def test_guided_learning_trainset_nonempty():
    from data.trainset_guided_learning import TRAINSET
    assert len(TRAINSET) >= 10


def test_guided_learning_examples_mark_inputs():
    from data.trainset_guided_learning import TRAINSET
    for ex in TRAINSET:
        keys = ex.inputs().keys()
        assert "question" in keys
        assert "past_attempts" in keys


def test_guided_learning_next_questions_end_with_qmark():
    from data.trainset_guided_learning import TRAINSET
    for ex in TRAINSET:
        assert ex.next_question.strip().endswith("?"), (
            f"next_question does not end with '?': {ex.question!r}"
        )


# ---------------------------------------------------------------------------
# deep_research_metric
# ---------------------------------------------------------------------------

def test_deep_research_metric_passes_on_full_overlap():
    ex = SimpleNamespace(citations="horcruxes lord-voldemort")
    pred = SimpleNamespace(citations="horcruxes lord-voldemort harry-potter")
    assert deep_research_metric(ex, pred) is True


def test_deep_research_metric_passes_on_majority_overlap():
    ex = SimpleNamespace(citations="a b c d")
    pred = SimpleNamespace(citations="a b z")   # 2/4 = exactly half
    assert deep_research_metric(ex, pred) is True


def test_deep_research_metric_fails_on_no_overlap():
    ex = SimpleNamespace(citations="horcruxes lord-voldemort")
    pred = SimpleNamespace(citations="hogwarts")
    assert deep_research_metric(ex, pred) is False


def test_deep_research_strict_metric_requires_all_citations():
    ex = SimpleNamespace(citations="horcruxes lord-voldemort")
    pred_missing = SimpleNamespace(citations="horcruxes")
    pred_all     = SimpleNamespace(citations="horcruxes lord-voldemort harry-potter")
    assert deep_research_strict_metric(ex, pred_missing) is False
    assert deep_research_strict_metric(ex, pred_all) is True


def test_deep_research_metric_passes_when_no_expected_citations():
    ex = SimpleNamespace(citations="")
    pred = SimpleNamespace(citations="anything")
    assert deep_research_metric(ex, pred) is True


# ---------------------------------------------------------------------------
# socratic_metric
# ---------------------------------------------------------------------------

def test_socratic_metric_passes_on_good_output():
    pred = SimpleNamespace(
        hint="Think about what happens when a wizard fears death above all else.",
        next_question="What kind of magic might split a soul?",
        explanation="The concept touches on a class of dark magic involving soul division.",
    )
    assert socratic_metric(None, pred) is True


def test_socratic_metric_fails_when_hint_contains_spoiler():
    pred = SimpleNamespace(
        hint="A horcrux is a container for a soul fragment.",
        next_question="What else?",
        explanation="ok",
    )
    assert socratic_metric(None, pred) is False


def test_socratic_metric_fails_when_next_question_not_a_question():
    pred = SimpleNamespace(
        hint="A perfectly reasonable hint that does not give anything away.",
        next_question="this is a statement.",
        explanation="fine explanation",
    )
    assert socratic_metric(None, pred) is False


def test_socratic_metric_fails_on_trivial_hint():
    pred = SimpleNamespace(
        hint="Think.",
        next_question="Why?",
        explanation="A reasonable explanation of the concept.",
    )
    assert socratic_metric(None, pred) is False


def test_socratic_score_rewards_quality():
    good = SimpleNamespace(
        hint="Think carefully about what Voldemort was trying to avoid and why.",
        next_question="What kind of act irreversibly damages a soul?",
        explanation="The concept involves fragmentation of the self via dark magic.",
    )
    bad = SimpleNamespace(
        hint="a horcrux is an object",
        next_question="statement not question",
        explanation="the answer is that it holds a piece of soul",
    )
    assert socratic_score(None, good) > socratic_score(None, bad)


# ---------------------------------------------------------------------------
# Debate trainset integrity
# ---------------------------------------------------------------------------

def test_debate_trainset_nonempty():
    from data.trainset_debate import TRAINSET
    assert len(TRAINSET) >= 10


def test_debate_trainset_citations_reference_valid_docs():
    from data.trainset_debate import TRAINSET, EVALSET
    for ex in TRAINSET + EVALSET:
        for cite in ex.citations.split():
            assert cite in VALID_DOC_IDS, (
                f"unknown citation {cite!r} in position: {ex.position!r}"
            )


def test_debate_examples_mark_position_as_input():
    from data.trainset_debate import TRAINSET
    for ex in TRAINSET:
        assert "position" in ex.inputs().keys()


# ---------------------------------------------------------------------------
# debate_metric
# ---------------------------------------------------------------------------

def test_debate_metric_passes_on_good_prediction():
    ex = SimpleNamespace(citations="severus-snape harry-potter")
    pred = SimpleNamespace(
        arguments_for="Snape risked his life as a double agent throughout the war.",
        arguments_against="Snape bullied students for years and acted from self-interest.",
        verdict="The canon supports his heroism more strongly, given the sacrifices made.",
        citations="severus-snape harry-potter albus-dumbledore",
    )
    assert debate_metric(ex, pred) is True


def test_debate_metric_fails_when_arguments_for_is_empty():
    ex = SimpleNamespace(citations="severus-snape")
    pred = SimpleNamespace(
        arguments_for="Short.",
        arguments_against="Snape was cruel to students throughout his tenure at Hogwarts.",
        verdict="The evidence is mixed but leans against.",
        citations="severus-snape",
    )
    assert debate_metric(ex, pred) is False


def test_debate_metric_fails_when_no_citation_overlap():
    ex = SimpleNamespace(citations="severus-snape harry-potter")
    pred = SimpleNamespace(
        arguments_for="Snape risked his life as a double agent throughout the war.",
        arguments_against="Snape bullied students for years and acted from self-interest.",
        verdict="The canon supports his heroism more strongly.",
        citations="hogwarts",  # no overlap with expected
    )
    assert debate_metric(ex, pred) is False


def test_debate_metric_passes_when_no_expected_citations():
    ex = SimpleNamespace(citations="")
    pred = SimpleNamespace(
        arguments_for="There are valid arguments in favour of this position.",
        arguments_against="There are also strong arguments against this position.",
        verdict="The evidence is balanced overall.",
        citations="anything",
    )
    assert debate_metric(ex, pred) is True


def test_debate_score_rewards_quality():
    good = SimpleNamespace(
        arguments_for="Snape risked everything as a double agent to protect Harry and the Order.",
        arguments_against="Snape bullied students systematically and prioritised personal obsession.",
        verdict="The weight of canon evidence supports Snape as ultimately heroic.",
        citations="severus-snape harry-potter",
    )
    bad = SimpleNamespace(
        arguments_for="ok",
        arguments_against="nope",
        verdict="",
        citations="",
    )
    assert debate_score(None, good) > debate_score(None, bad)

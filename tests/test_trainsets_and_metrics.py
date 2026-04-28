"""Tests for data/trainset_*.py and context_harness/metrics.py"""

from types import SimpleNamespace
import pytest

from context_harness.metrics import (
    deep_research_metric, deep_research_strict_metric,
    socratic_metric, socratic_score,
    debate_metric, debate_score,
    satirical_podcast_metric, satirical_podcast_score,
    perspective_shift_metric,
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


# ---------------------------------------------------------------------------
# Satirical Podcast trainset integrity
# ---------------------------------------------------------------------------

def test_satirical_podcast_trainset_nonempty():
    from data.trainset_satirical_podcast import TRAINSET
    assert len(TRAINSET) >= 15


def test_satirical_podcast_trainset_citations_reference_valid_docs():
    from data.trainset_satirical_podcast import TRAINSET, EVALSET
    for ex in TRAINSET + EVALSET:
        for cite in ex.citations.split():
            assert cite in VALID_DOC_IDS, (
                f"unknown citation {cite!r} in topic: {ex.topic!r}"
            )


def test_satirical_podcast_examples_mark_both_inputs():
    from data.trainset_satirical_podcast import TRAINSET
    for ex in TRAINSET:
        keys = ex.inputs().keys()
        assert "topic" in keys, f"'topic' not marked as input in: {ex.topic!r}"
        assert "modern_angle" in keys, f"'modern_angle' not marked as input in: {ex.topic!r}"


def test_satirical_podcast_evalset_disjoint_from_train():
    from data.trainset_satirical_podcast import TRAINSET, EVALSET
    train_topics = {ex.topic for ex in TRAINSET}
    eval_topics  = {ex.topic for ex in EVALSET}
    assert train_topics.isdisjoint(eval_topics)


# ---------------------------------------------------------------------------
# satirical_podcast_metric
# ---------------------------------------------------------------------------

_GOOD_TRANSCRIPT = (
    "Lavender: So I ordered the invisibility cloak on WizAmazon — Prime Owl, supposedly two-day delivery.\n"
    "Parvati: Two days? My owl took three generations to arrive. The original owl retired halfway through.\n"
    "Lavender: And when it finally got here the cloak was listed as 'used — good condition'. I can SEE it!\n"
    "Parvati: That's not how invisibility works, Lavender. That's fraud.\n"
    "Lavender: I left a one-star review and they sent a Howler to my address."
)


def test_satirical_podcast_metric_passes_on_good_prediction():
    ex = SimpleNamespace(citations="deathly-hallows")
    pred = SimpleNamespace(
        transcript=_GOOD_TRANSCRIPT,
        comedic_tension="A cloak that makes you invisible cannot be inspected for condition, making quality control literally impossible.",
        citations="deathly-hallows hogwarts",
    )
    assert satirical_podcast_metric(ex, pred) is True


def test_satirical_podcast_metric_fails_when_transcript_too_short():
    ex = SimpleNamespace(citations="hogwarts")
    pred = SimpleNamespace(
        transcript="Host: Hi. Guest: Hello.",
        comedic_tension="Magic meets modernity in an amusing way.",
        citations="hogwarts",
    )
    assert satirical_podcast_metric(ex, pred) is False


def test_satirical_podcast_metric_fails_when_too_few_dialogue_lines():
    ex = SimpleNamespace(citations="hogwarts")
    pred = SimpleNamespace(
        # Long but no "Speaker: text" dialogue structure
        transcript="A" * 200,
        comedic_tension="Magic meets modernity in an amusing way that creates tension.",
        citations="hogwarts",
    )
    assert satirical_podcast_metric(ex, pred) is False


def test_satirical_podcast_metric_fails_when_no_citations():
    ex = SimpleNamespace(citations="deathly-hallows")
    pred = SimpleNamespace(
        transcript=_GOOD_TRANSCRIPT,
        comedic_tension="A cloak that makes you invisible cannot be quality-checked.",
        citations="",
    )
    assert satirical_podcast_metric(ex, pred) is False


def test_satirical_podcast_metric_fails_when_comedic_tension_trivial():
    ex = SimpleNamespace(citations="hogwarts")
    pred = SimpleNamespace(
        transcript=_GOOD_TRANSCRIPT,
        comedic_tension="Funny.",
        citations="hogwarts",
    )
    assert satirical_podcast_metric(ex, pred) is False


def test_satirical_podcast_score_rewards_quality():
    good = SimpleNamespace(
        transcript=_GOOD_TRANSCRIPT,
        comedic_tension="A cloak that makes you invisible cannot be inspected for condition.",
        citations="deathly-hallows hogwarts",
    )
    bad = SimpleNamespace(
        transcript="ok",
        comedic_tension="",
        citations="",
    )
    assert satirical_podcast_score(None, good) > satirical_podcast_score(None, bad)


# ---------------------------------------------------------------------------
# perspective_shift_metric — multi-turn greeting guard + character_response
# substance check (added 2026-04-23 per PR #26 review findings)
# ---------------------------------------------------------------------------

def _good_pshift_pred(**overrides):
    """A prediction that passes all perspective_shift_metric length checks.
    Override specific fields per-test."""
    base = dict(
        character_principle="A principle at least fifty characters long referencing specific canon.",
        applied_insight="An actionable insight that exceeds eighty characters by design, giving a specific stance the user can take this week about their situation.",
        reasoning="Reasoning that bridges character to scenario in at least fifty characters of prose.",
        character_response=(
            "First-person response from the character, at least one hundred "
            "characters of substance, weaving canon naturally without "
            "greeting openers or section labels."
        ),
        citations="[character/section-001]",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_perspective_shift_passes_well_formed_single_turn():
    """Baseline: a full prediction with empty chat_history should pass."""
    example = SimpleNamespace(chat_history="")
    pred = _good_pshift_pred()
    assert perspective_shift_metric(example, pred) is True


def test_perspective_shift_rejects_short_character_response():
    """character_response < 100 chars fails — the synthesis field must be
    substantive. Regression guard: pre-review, the metric ignored this
    field entirely and would pass demos with bad voice."""
    example = SimpleNamespace(chat_history="")
    pred = _good_pshift_pred(character_response="Short reply.")
    assert perspective_shift_metric(example, pred) is False


def test_perspective_shift_rejects_greeting_opener_on_multi_turn():
    """The load-bearing regression guard for this PR. When chat_history is
    non-empty, character_response starting with 'Oh, hello there!' (or
    similar cold-start greetings) must fail the metric so BootstrapFewShot
    never promotes such a response as a demo. Regression target:
    2026-04-21 Luna bug."""
    example = SimpleNamespace(chat_history="[1] User: I'm dealing with imposter syndrome.\n     Tutor: ...")
    for opener in [
        "Oh, hello there! I'm delighted you've come to see me. Let me tell you about...",
        "Hi there, friend! Today we'll discuss imposter syndrome from Luna's perspective and...",
        "Hello, welcome to our conversation. I wanted to reintroduce myself before we continue and...",
        "Welcome back! It's wonderful to see you again. Now, about your question on imposter...",
    ]:
        pred = _good_pshift_pred(character_response=opener)
        assert perspective_shift_metric(example, pred) is False, \
            f"Greeting opener should be rejected: {opener!r}"


def test_perspective_shift_allows_greeting_opener_on_cold_start():
    """When chat_history is empty, the greeting guard does NOT fire — a
    character greeting a first-time user is legitimate. Only multi-turn
    context triggers the no-greeting rule."""
    example = SimpleNamespace(chat_history="")
    pred = _good_pshift_pred(
        character_response=(
            "Oh, hello there! I sense something weighing on you. Let me share "
            "what my own life has taught me about this sort of question. "
            "Sometimes the Wrackspurts cloud our judgement in these moments."
        ),
    )
    assert perspective_shift_metric(example, pred) is True


def test_perspective_shift_still_rejects_missing_citations():
    """Existing check preserved: no citations → fail."""
    example = SimpleNamespace(chat_history="")
    pred = _good_pshift_pred(citations="")
    assert perspective_shift_metric(example, pred) is False
    pred = _good_pshift_pred(citations="none")
    assert perspective_shift_metric(example, pred) is False


def test_perspective_shift_multi_turn_without_greeting_passes():
    """Positive case for multi-turn: chat_history present + response
    continues without greeting → passes."""
    example = SimpleNamespace(chat_history="[1] User: I'm tired.\n     Tutor: ...")
    pred = _good_pshift_pred(
        character_response=(
            "Tired. Yes. I know that word intimately. Hear me carefully: "
            "usefulness does not require you to feel capable. It requires "
            "only that you do one small thing tomorrow, unnoticed, that you "
            "would have done for them. That is what I had. It was enough."
        ),
    )
    assert perspective_shift_metric(example, pred) is True


def test_perspective_shift_rejects_deflection_on_reflection_request():
    """When the user asks to summarize their situation, a response that
    ignores what they shared (zero content overlap with chat_history)
    should fail. This is the bug this feature fixes."""
    example = SimpleNamespace(
        scenario="can you summarize my situation?",
        chat_history=(
            "[1] User: I have a safe job offer and a risky creative path.\n"
            "     Tutor: What does the safe path cost you in ten years?\n"
            "[2] User: Probably just regret. The creative path could cost "
            "me stability."
        ),
    )
    pred = _good_pshift_pred(
        character_response=(
            "Ah, courage is a curious thing. One must always remember that "
            "it is our choices that show what we truly are, far more than "
            "our abilities. The world offers many roads and each has its "
            "own particular enchantment. What matters is the heart."
        ),
    )
    assert perspective_shift_metric(example, pred) is False


def test_perspective_shift_passes_reflection_with_content_echo():
    """When the user asks for a summary and the response restates their
    scenario using content words from chat_history, the metric passes."""
    example = SimpleNamespace(
        scenario="can you summarize my situation?",
        chat_history=(
            "[1] User: I have a safe job offer and a risky creative path.\n"
            "     Tutor: What does the safe path cost you in ten years?\n"
            "[2] User: Probably just regret. The creative path could cost "
            "me stability."
        ),
    )
    pred = _good_pshift_pred(
        character_response=(
            "Of course. You came to me torn between a safe career and a "
            "creative path that excites you but offers no guarantees. When "
            "I asked what the safe road costs, you said regret — and when "
            "you looked at the creative road, you said stability. I turned "
            "down the most powerful position in our world because I knew "
            "what it would quietly take from me. Which of those two costs "
            "are you more practised at surviving?"
        ),
    )
    assert perspective_shift_metric(example, pred) is True


def test_perspective_shift_reflection_multi_turn_real_length():
    """Regression test: a long multi-turn conversation where the user asks
    'can you summarize my situation?'. A deflecting response that ignores
    the user's content should fail; a reflecting response that echoes key
    details should pass."""
    chat_history = (
        "[1] User: My colleague praised my work publicly but takes credit "
        "for it in private meetings with management. I confronted them "
        "and they called me paranoid.\n"
        "     Tutor: What happened when you confronted them?\n"
        "[2] User: They said I was imagining things and that credit "
        "doesn't matter.\n"
        "     Tutor: Has anyone else noticed the pattern?\n"
        "[3] User: My teammate mentioned it once but didn't want to "
        "get involved.\n"
        "     Tutor: What did your manager say when you raised it?\n"
        "[4] User: I haven't raised it with my manager yet. I'm afraid "
        "of looking petty.\n"
        "     Tutor: What would change if your manager knew?"
    )
    example = SimpleNamespace(
        scenario="can you summarize my situation?",
        chat_history=chat_history,
    )

    # BAD: deflects with HP analogies, never restates the user's story
    deflecting_pred = _good_pshift_pred(
        character_response=(
            "Well, sometimes the most important thing is to trust your "
            "instincts. When Ron and I had our difficulties, I learned "
            "that clarity comes from within, not from others. The heart "
            "knows what the mind refuses to accept. You must be brave "
            "enough to face the truth, whatever it may be. What do you "
            "think your next step should be?"
        ),
    )
    assert perspective_shift_metric(example, deflecting_pred) is False, \
        "Deflecting response should fail — no content from user's story echoed"

    # GOOD: restates the user's situation using their own content words
    reflecting_pred = _good_pshift_pred(
        character_response=(
            "Let me lay it out plainly. Your colleague takes credit for "
            "your work in private meetings while praising you publicly. "
            "When you confronted them, they called you paranoid. Your "
            "teammate noticed but won't get involved, and you haven't "
            "raised it with your manager because you're afraid of looking "
            "petty. I understand that fear — I spent years watching others "
            "take credit for work that wasn't theirs. The question is "
            "whether staying quiet costs you more than speaking up."
        ),
    )
    assert perspective_shift_metric(example, reflecting_pred) is True, \
        "Reflecting response should pass — echoes user's content words"

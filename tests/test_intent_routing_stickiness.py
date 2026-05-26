"""
Intent-router stickiness — decision-helper unit tests
=====================================================
``_decide_effective_mode(routed_mode, router_confidence, prior_mode,
prior_character)`` is the pure function that picks the effective mode
for a turn. The rules:

  1. No prior_mode → trust the router.
  2. prior_mode == "none" → router gets a fresh vote.
  3. routed_mode == prior_mode → no change.
  4. character-lock: prior was perspective_shift + non-default character,
     routed to a non-perspective_shift mode → stay in perspective_shift.
  5. high-confidence different mode → switch (router-override).
  6. otherwise → stick to prior_mode.

Returns ``(effective_mode, source)`` where source ∈ {"router", "sticky",
"router-override", "character-lock"}. Pure: no IO, no globals — every
input is an explicit argument, so unit tests don't need fixtures or
mocks.

Backs SPEC.md acceptance criteria 1–5 + the 6-row decision table in
tasks/plan.md.
"""

from __future__ import annotations

import pytest

from api.main import _decide_effective_mode


@pytest.mark.parametrize(
    "routed_mode, confidence, prior_mode, expected_mode, expected_source",
    [
        # 1. Turn 1 / no conversation_id → trust the router
        pytest.param(
            "open_analysis", "high", None,
            "open_analysis", "router",
            id="no_prior_mode_uses_router_directly",
        ),
        # 2. Continuation in chat mode + ambiguous follow-up classified as
        #    off-topic → keep prior mode (stickiness)
        pytest.param(
            "none", "high", "open_analysis",
            "open_analysis", "sticky",
            id="continuation_inherits_prior_mode_on_off_topic_classification",
        ),
        # 3. Continuation + low-confidence different mode → keep prior
        pytest.param(
            "deep_research", "low", "open_analysis",
            "open_analysis", "sticky",
            id="low_confidence_router_does_not_override",
        ),
        # 4. Continuation + high-confidence different mode → honor switch
        pytest.param(
            "debate", "high", "open_analysis",
            "debate", "router-override",
            id="high_confidence_switch_overrides_stickiness",
        ),
        # 5. Continuation + router agrees with prior → no-op (counts as router)
        pytest.param(
            "open_analysis", "medium", "open_analysis",
            "open_analysis", "router",
            id="router_agrees_with_prior_mode",
        ),
        # 6. Prior was off-topic → router gets a fresh vote (no stickiness on
        #    "none" prior; user might be re-engaging on a new topic)
        pytest.param(
            "deep_research", "high", "none",
            "deep_research", "router",
            id="prior_off_topic_lets_router_pick_fresh",
        ),
    ],
)
def test_decide_effective_mode(
    routed_mode, confidence, prior_mode, expected_mode, expected_source,
):
    actual_mode, actual_source = _decide_effective_mode(
        routed_mode=routed_mode,
        router_confidence=confidence,
        prior_mode=prior_mode,
    )
    assert actual_mode == expected_mode, (
        f"expected effective_mode={expected_mode!r}, got {actual_mode!r}"
    )
    assert actual_source == expected_source, (
        f"expected source={expected_source!r}, got {actual_source!r}"
    )


def test_decide_effective_mode_handles_none_routed_mode_gracefully():
    """Defensive: if the router somehow returned None (e.g. parse failure),
    fall back to prior_mode if available, else propagate None."""
    # No prior, no routed → caller's problem; we return None untouched
    mode, source = _decide_effective_mode(
        routed_mode=None, router_confidence=None, prior_mode=None,
    )
    assert mode is None
    assert source == "router"

    # Prior set, routed None → stickiness saves us
    mode, source = _decide_effective_mode(
        routed_mode=None, router_confidence=None, prior_mode="open_analysis",
    )
    assert mode == "open_analysis"
    assert source == "sticky"


# ---------- Character-lock rule (Rule 4) ----------------------------------
#
# Reproduces the 2026-05-25 production bug: a Luna chat (perspective_shift +
# luna-lovegood) where turn 3's analytical-sounding follow-up ("so I might
# likely to have the capacity to reconnect if I can understand the WHOLE
# picture of them") got classified high-confidence as open_analysis. Without
# character-lock, Rule 5 (router-override) fires; with it, we stay in voice.

@pytest.mark.parametrize(
    "routed_mode, confidence, prior_character, expected_mode, expected_source",
    [
        # Bug repro: high-confidence open_analysis routing in active Luna chat
        # → character-lock holds the mode.
        pytest.param(
            "open_analysis", "high", "luna-lovegood",
            "perspective_shift", "character-lock",
            id="active_character_chat_locks_against_high_conf_open_analysis",
        ),
        # Same lock applies to other cross-mode high-confidence routes
        # (debate, satirical_podcast, deep_research) — any non-perspective
        # destination is suppressed when a real character is bound.
        pytest.param(
            "debate", "high", "sirius-black",
            "perspective_shift", "character-lock",
            id="active_character_chat_locks_against_debate_route",
        ),
        pytest.param(
            "satirical_podcast", "high", "minerva-mcgonagall",
            "perspective_shift", "character-lock",
            id="active_character_chat_locks_against_satire_route",
        ),
        # Low-confidence routes hit stickiness BEFORE character-lock, but
        # they still resolve to perspective_shift either way — character-lock
        # just labels the source for telemetry / future debugging.
        pytest.param(
            "open_analysis", "low", "luna-lovegood",
            "perspective_shift", "character-lock",
            id="active_character_chat_locks_low_conf_route_too",
        ),
        # Legacy "Dumbledore" sentinel is treated as unset — older clients
        # that silently defaulted to Dumbledore shouldn't accidentally lock
        # modes mid-conversation. Falls through to normal stickiness rules.
        pytest.param(
            "open_analysis", "high", "Dumbledore",
            "open_analysis", "router-override",
            id="dumbledore_default_does_not_trigger_character_lock",
        ),
        # No prior character (just perspective_shift with empty character)
        # → no lock, normal rules apply.
        pytest.param(
            "open_analysis", "high", None,
            "open_analysis", "router-override",
            id="no_prior_character_skips_lock",
        ),
        # Router agrees with prior perspective_shift → Rule 3 wins before
        # character-lock is consulted. Source stays "router", not "lock".
        pytest.param(
            "perspective_shift", "high", "luna-lovegood",
            "perspective_shift", "router",
            id="router_agrees_no_lock_needed",
        ),
        # Router has no opinion (routed_mode="none") inside an active Luna
        # chat → lock does NOT fire; falls through to Rule 6 sticky. Same
        # outcome (perspective_shift) but the source is "sticky" because
        # we're not overriding any positive route — telemetry stays honest.
        pytest.param(
            "none", "high", "luna-lovegood",
            "perspective_shift", "sticky",
            id="no_opinion_route_falls_through_to_sticky_not_lock",
        ),
        # Whitespace-only character slug must NOT pass the lock. Today's
        # store layer trims input but the predicate has to be robust
        # regardless — _is_bound_character() runs the same .strip() check
        # the carry-forward heuristic uses, so the two can't desync.
        pytest.param(
            "open_analysis", "high", "   ",
            "open_analysis", "router-override",
            id="whitespace_only_character_does_not_lock",
        ),
    ],
)
def test_character_lock_rule(
    routed_mode, confidence, prior_character, expected_mode, expected_source,
):
    actual_mode, actual_source = _decide_effective_mode(
        routed_mode=routed_mode,
        router_confidence=confidence,
        prior_mode="perspective_shift",
        prior_character=prior_character,
    )
    assert actual_mode == expected_mode
    assert actual_source == expected_source


def test_character_lock_only_triggers_when_prior_was_perspective_shift():
    """A Luna character bound to a prior open_analysis turn (somehow) must
    NOT lock — character-lock is gated on the prior MODE being a real
    character chat, not just any prior turn that happened to carry a
    character slug."""
    mode, source = _decide_effective_mode(
        routed_mode="debate",
        router_confidence="high",
        prior_mode="open_analysis",
        prior_character="luna-lovegood",  # weird state, shouldn't lock
    )
    assert mode == "debate"
    assert source == "router-override"

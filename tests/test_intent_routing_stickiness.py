"""
Intent-router stickiness — decision-helper unit tests
=====================================================
``_decide_effective_mode(routed_mode, router_confidence, prior_mode)`` is the
pure function that picks the effective mode for a turn. The rules:

  1. No prior_mode → trust the router.
  2. prior_mode == "none" → router gets a fresh vote.
  3. routed_mode == prior_mode → no change.
  4. high-confidence different mode → switch (router-override).
  5. otherwise → stick to prior_mode.

Returns ``(effective_mode, source)`` where source ∈ {"router", "sticky",
"router-override"}. Pure: no IO, no globals — every input is an explicit
argument, so unit tests don't need fixtures or mocks.

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

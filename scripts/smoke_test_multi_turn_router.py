"""
Multi-turn intent-router smoke test
====================================
Drives a 4-turn conversation against a locally-running ``api.main`` server
to verify that stickiness holds on ambiguous follow-ups and that
high-confidence intent switches still override.

This is the integration counterpart to ``tests/test_api_intent_stickiness.py``
(which uses TestClient + DummyLM). This script hits a *real* Gemini-backed
server, so it costs ~4 LLM-routing-calls + ~4 mode-dispatch calls (~$0.005
total at flash-lite). Skipped in CI without keys; runnable locally.

Usage
-----
Terminal 1::

    source .env
    .venv/bin/python -m uvicorn api.main:app --port 8000

Terminal 2::

    .venv/bin/python -m scripts.smoke_test_multi_turn_router

The script asserts on ``routed_mode`` per turn and prints PASS/FAIL.
Exit code 0 on full pass, 1 on any failure.
"""

from __future__ import annotations

import json
import sys
import time

import requests


SERVER_URL = "http://127.0.0.1:8000/ask"
TIMEOUT_SECS = 60


def _post(payload: dict) -> dict:
    r = requests.post(SERVER_URL, json=payload, timeout=TIMEOUT_SECS)
    r.raise_for_status()
    return r.json()


def _check(turn_label: str, body: dict, *, expected_routed_mode: str,
           expected_effective_mode: str | None = None) -> bool:
    """Assert turn outcome. Returns True iff all checks pass."""
    routed = body.get("routed_mode")
    print(f"\n=== {turn_label} ===")
    print(f"  routed_mode      = {routed!r}")
    print(f"  router_confidence= {body.get('router_confidence')!r}")
    print(f"  answer (first 120): {(body.get('answer') or '')[:120]}...")

    ok = routed == expected_routed_mode
    if not ok:
        print(f"  ✗ FAIL: expected routed_mode = {expected_routed_mode!r}")
        return False

    # The /ask endpoint doesn't currently surface effective_mode in the
    # response body — only router output. We rely on the answer shape +
    # cost to confirm that the dispatched mode matched expectations.
    print("  ✓ PASS")
    return True


def main() -> int:
    cid = f"smoke-router-{int(time.time())}"
    failures: list[str] = []

    print(f"Conversation ID: {cid}")
    print(f"Hitting {SERVER_URL} (timeout={TIMEOUT_SECS}s per turn)\n")

    # --- Turn 1: clear analysis intent ---
    body1 = _post({
        "question": "What does Hogwarts say about British boarding-school culture?",
        "mode": "auto",
        "conversation_id": cid,
    })
    if not _check(
        "Turn 1: explicit analysis question", body1,
        expected_routed_mode="open_analysis",
    ):
        failures.append("turn1")

    # --- Turn 2: ambiguous continuation — must STICK to open_analysis ---
    body2 = _post({
        "question": "cool",
        "mode": "auto",
        "conversation_id": cid,
    })
    # The router may legitimately classify "cool" as "none" (off-topic) —
    # that's the whole point. We still expect dispatch to land in
    # open_analysis (the prior mode). The body's routed_mode reflects what
    # the router emitted, not the dispatch decision; the dispatch is sticky.
    print(f"\n=== Turn 2: ambiguous follow-up ('cool') ===")
    print(f"  routed_mode      = {body2.get('routed_mode')!r}")
    print(f"  router_confidence= {body2.get('router_confidence')!r}")
    print(f"  answer (first 120): {(body2.get('answer') or '')[:120]}...")
    # If routed_mode != "none", the router happened to stay sticky on its own.
    # Either way, the dispatch should NOT be the off-topic capability text.
    capability_marker = "I'm a Harry Potter lore agent"
    if capability_marker in (body2.get("answer") or ""):
        print(f"  ✗ FAIL: turn 2 returned the off-topic capability screen — "
              f"stickiness did not fire")
        failures.append("turn2")
    else:
        print("  ✓ PASS (dispatched to a real mode, not the help screen)")

    # --- Turn 3: another short continuation ---
    body3 = _post({
        "question": "tell me more",
        "mode": "auto",
        "conversation_id": cid,
    })
    print(f"\n=== Turn 3: another continuation ('tell me more') ===")
    print(f"  routed_mode      = {body3.get('routed_mode')!r}")
    print(f"  answer (first 120): {(body3.get('answer') or '')[:120]}...")
    if capability_marker in (body3.get("answer") or ""):
        print(f"  ✗ FAIL: turn 3 collapsed to capability screen")
        failures.append("turn3")
    else:
        print("  ✓ PASS")

    # --- Turn 4: explicit mode-switch intent ---
    body4 = _post({
        "question": "Actually, let's debate this — was Voldemort ever redeemable?",
        "mode": "auto",
        "conversation_id": cid,
    })
    if not _check(
        "Turn 4: explicit debate switch", body4,
        expected_routed_mode="debate",
    ):
        failures.append("turn4")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED — {len(failures)} turn(s): {', '.join(failures)}")
        return 1
    print("ALL 4 TURNS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.RequestException as e:
        print(f"\nFAILED — could not reach {SERVER_URL}: {e}")
        print("Is the server running? `uvicorn api.main:app --port 8000`")
        sys.exit(2)

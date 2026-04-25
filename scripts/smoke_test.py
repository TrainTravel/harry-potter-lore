"""
Post-deploy smoke test with LLM judge.

Hits the deployed /ask endpoint with a multi-turn perspective_shift conversation,
then uses an LLM judge to verify the response actually reflects back the user's
content — the failure mode we discovered after merging the reflection fix.

Usage:
    # Against Railway
    .venv/bin/python scripts/smoke_test.py --base-url https://your-app.railway.app

    # Against localhost (default)
    .venv/bin/python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

import litellm
import requests


# ---------------------------------------------------------------------------
# Scenario: perspective_shift reflection
# ---------------------------------------------------------------------------

TURNS = [
    {
        "question": (
            "I've been through a lot of hardship growing up and I feel like "
            "it's made me bitter and closed off. I push people away even when "
            "they're trying to help."
        ),
        "mode": "perspective_shift",
        "character": "Dumbledore",
    },
    {
        "question": (
            "yeah especially at work. my coworkers think I'm cold but really "
            "I'm just protecting myself"
        ),
        "mode": "perspective_shift",
        "character": "Dumbledore",
    },
    {
        "question": "can you summarize my situation?",
        "mode": "perspective_shift",
        "character": "Dumbledore",
    },
]

JUDGE_PROMPT = """\
You are evaluating a character-AI's response. The user had a multi-turn \
conversation and then asked the character to summarize their situation.

Conversation:
- User turn 1: "{turn1}"
- User turn 2: "{turn2}"

User's request: "can you summarize my situation?"

Character's response:
{response}

Does the character's response restate specific details from the user's \
situation (hardship, bitter, closed off, pushing people away, coworkers \
thinking they're cold, protecting themselves)? Answer YES or NO, then one \
sentence explaining why."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_health(base_url: str, timeout: int = 30) -> None:
    print(f"GET {base_url}/health ... ", end="", flush=True)
    r = requests.get(f"{base_url}/health", timeout=timeout)
    r.raise_for_status()
    body = r.json()
    assert body.get("status") == "ok", f"unexpected health status: {body}"
    print("ok")


def send_turn(base_url: str, body: dict, timeout: int = 120) -> dict:
    r = requests.post(f"{base_url}/ask", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def judge(turn1: str, turn2: str, response: str, model: str) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT.format(turn1=turn1, turn2=turn2, response=response)
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    verdict = resp.choices[0].message.content.strip()
    passed = verdict.upper().startswith("YES")
    return passed, verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Post-deploy smoke test with LLM judge")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--judge-model", default="gemini/gemini-2.5-flash-lite",
                    help="LiteLLM model string for the judge call")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-request timeout in seconds")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds between /ask calls")
    args = ap.parse_args()

    # 1. Health check
    check_health(args.base_url, timeout=args.timeout)

    # 2. Multi-turn conversation
    conversation_id = str(uuid.uuid4())
    answers: list[str] = []

    for i, turn in enumerate(TURNS, 1):
        body = {**turn, "conversation_id": conversation_id}
        print(f"Turn {i}/3: {turn['question'][:60]}... ", end="", flush=True)
        t0 = time.time()
        resp = send_turn(args.base_url, body, timeout=args.timeout)
        dt = time.time() - t0
        answer = resp.get("answer", "")
        answers.append(answer)
        print(f"{dt:.1f}s  ({len(answer)} chars)")
        if i < len(TURNS):
            time.sleep(args.delay)

    # 3. LLM judge on turn-3 response
    print(f"\nJudging with {args.judge_model}... ", end="", flush=True)
    passed, verdict = judge(
        turn1=TURNS[0]["question"],
        turn2=TURNS[1]["question"],
        response=answers[2],
        model=args.judge_model,
    )

    status = "PASS" if passed else "FAIL"
    print(f"{status}")
    print(f"\nJudge verdict: {verdict}")
    print(f"\nTurn-3 response:\n{answers[2][:500]}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""
End-to-end agent eval
=====================
Runs the ``LoreAgent`` against every question in ``evals.questions`` and
scores:
  - answer_correctness  (LLM-as-judge via Gemini Flash-Lite)
  - retrieval_hit       (did the agent's search_lore call return an expected doc?)
  - tokens_used
  - tool_calls
  - turns
  - cost_usd

Run:
    .venv/bin/python -m evals.eval_agent

Tip: pass ``--limit N`` to run against only the first N questions while debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Set

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

from evals.agent import LoreAgent, AgentRunResult, MODEL
from evals.questions import QUESTIONS, EvalQuestion


# Gemini pricing (USD per 1M tokens, AI Studio paid tier). Free tier has
# generous daily quotas; these are for cost telemetry only.
PRICING = {
    "gemini-2.5-flash":      {"in": 0.30, "out": 2.50},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-2.5-pro":        {"in": 1.25, "out": 10.00},
}

JUDGE_MODEL = "gemini-2.5-flash-lite"


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING.get(model, {"in": 0.0, "out": 0.0})
    return (in_tok * p["in"] + out_tok * p["out"]) / 1_000_000


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict grader. Compare the AGENT ANSWER to the REFERENCE ANSWER. "
    "Reply with a single JSON object: {\"correct\": true|false, \"reason\": \"<short>\"}. "
    "Mark correct if the agent's answer contains the key facts from the reference "
    "(minor phrasing differences are OK). For distractor questions, mark correct only "
    "if the agent explicitly admits it doesn't know."
)


def judge(
    client: "genai.Client",
    question: str,
    agent_answer: str,
    reference: str,
    kind: str,
) -> dict:
    prompt = (
        f"QUESTION: {question}\n"
        f"KIND: {kind}\n"
        f"REFERENCE ANSWER: {reference}\n"
        f"AGENT ANSWER: {agent_answer}\n\n"
        "Respond with JSON only."
    )
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM,
            max_output_tokens=256,
            response_mime_type="application/json",
        ),
    )
    text = (resp.text or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(m.group(0) if m else text)
    except Exception:
        parsed = {"correct": False, "reason": f"judge parse failure: {text[:120]}"}
    usage = resp.usage_metadata
    judge_in = (usage.prompt_token_count or 0) if usage else 0
    judge_out = (usage.candidates_token_count or 0) if usage else 0
    parsed["_judge_cost_usd"] = cost_usd(JUDGE_MODEL, judge_in, judge_out)
    return parsed


# ---------------------------------------------------------------------------
# Retrieval hit detection from the transcript
# ---------------------------------------------------------------------------

def retrieval_hit(result: AgentRunResult, expected_docs: Set[str]) -> Optional[bool]:
    """Did the agent's search_lore call(s) return any expected doc?

    Returns None for distractor questions (expected_docs empty). Otherwise
    True if any expected doc_id appeared in any tool_result, False if the
    agent called the tool but never retrieved an expected doc, and False
    if the agent never called the tool at all.
    """
    if not expected_docs:
        return None
    retrieved: Set[str] = set()
    for entry in result.transcript:
        for tr in entry.get("tool_results", []) or []:
            for p in tr.get("result", {}).get("passages", []):
                doc_id = p.get("doc_id")
                if doc_id:
                    retrieved.add(doc_id)
    return bool(retrieved & expected_docs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class QRow:
    id: str
    kind: str
    correct: bool
    retrieval_hit: Optional[bool]
    answer: str
    reason: str
    turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    agent_cost_usd: float
    judge_cost_usd: float
    latency_s: float


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default="evals/results.json")
    args = parser.parse_args()

    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: set GOOGLE_API_KEY (free at https://aistudio.google.com/apikey)",
              file=sys.stderr)
        sys.exit(2)

    questions: List[EvalQuestion] = QUESTIONS[: args.limit] if args.limit else QUESTIONS

    client = genai.Client()
    agent = LoreAgent(client=client)

    rows: List[QRow] = []
    for q in questions:
        print(f"\n>>> [{q.id} / {q.kind}] {q.question}")
        t0 = time.perf_counter()
        result = agent.ask(q.question, verbose=False)
        latency = time.perf_counter() - t0

        verdict = judge(
            client=client,
            question=q.question,
            agent_answer=result.answer,
            reference=q.reference_answer,
            kind=q.kind,
        )

        row = QRow(
            id=q.id, kind=q.kind,
            correct=bool(verdict.get("correct")),
            retrieval_hit=retrieval_hit(result, q.expected_docs),
            answer=result.answer,
            reason=str(verdict.get("reason", "")),
            turns=result.turns,
            tool_calls=result.tool_calls,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_tokens=result.cache_read_tokens,
            agent_cost_usd=cost_usd(MODEL, result.input_tokens, result.output_tokens),
            judge_cost_usd=float(verdict.get("_judge_cost_usd", 0.0)),
            latency_s=latency,
        )
        rows.append(row)
        print(f"    correct={row.correct}  turns={row.turns}  "
              f"tokens={row.input_tokens}+{row.output_tokens}  "
              f"cost=${row.agent_cost_usd:.4f}")
        print(f"    answer: {row.answer[:160]}")

    # --- summary ---
    total_agent_cost = sum(r.agent_cost_usd for r in rows)
    total_judge_cost = sum(r.judge_cost_usd for r in rows)
    total = len(rows)
    correct = sum(r.correct for r in rows)
    by_kind = {}
    for r in rows:
        k = r.kind
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += int(r.correct)
        by_kind[k][1] += 1

    hits = [r for r in rows if r.retrieval_hit is not None]
    hit_rate = sum(r.retrieval_hit for r in hits) / max(len(hits), 1)

    print("\n========= SUMMARY =========")
    print(f"  correct:          {correct}/{total} ({correct/total:.1%})")
    for k, (c, n) in sorted(by_kind.items()):
        print(f"    {k:11s}      {c}/{n}")
    print(f"  retrieval hit:    {sum(r.retrieval_hit for r in hits)}/{len(hits)} ({hit_rate:.1%})")
    print(f"  agent tokens in:  {sum(r.input_tokens for r in rows):>7d}")
    print(f"  agent tokens out: {sum(r.output_tokens for r in rows):>7d}")
    print(f"  cache reads:      {sum(r.cache_read_tokens for r in rows):>7d}")
    print(f"  mean tool calls:  {sum(r.tool_calls for r in rows)/total:.2f}")
    print(f"  mean turns:       {sum(r.turns for r in rows)/total:.2f}")
    print(f"  agent cost:       ${total_agent_cost:.4f}")
    print(f"  judge cost:       ${total_judge_cost:.4f}")
    print(f"  TOTAL cost:       ${total_agent_cost + total_judge_cost:.4f}")

    with open(args.out, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print(f"\n  wrote per-question rows to {args.out}")


if __name__ == "__main__":
    main()

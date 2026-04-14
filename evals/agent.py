"""
Minimal Gemini lore agent
=========================
Wires the ``context_harness`` library into an actual agent loop on Gemini.

  - ``ContextWindow`` holds system prompt + turn history with a token budget
  - ``RAGPipeline`` is exposed to Gemini as the ``search_lore`` tool
  - Manual agentic loop (not the ADK) so we can observe token usage,
    tool calls, and context state at every step

Run a quick smoke test:
    .venv/bin/python -m evals.agent "Who killed Dumbledore?"

Env:
    GOOGLE_API_KEY must be set (free tier available at https://aistudio.google.com/apikey).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from google.genai import errors as genai_errors


# ---------------------------------------------------------------------------
# Retry helper — see README discussion; port to cats-retry in scala-harness/
# ---------------------------------------------------------------------------

T = TypeVar("T")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Run fn() with exponential backoff + jitter on transient Gemini errors.

    Retries on 429 / 5xx; raises immediately on 4xx (other than 429) and
    non-API exceptions. Worst-case total wait with defaults is ~62s:
    1 + 2 + 4 + 8 + 16 + up to 30 on the last delay, plus jitter.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except genai_errors.APIError as e:
            status = getattr(e, "code", None)
            if status not in RETRYABLE_STATUS or attempt == max_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1), max_delay)
            print(f"[retry {attempt}/{max_attempts - 1}] {status} — sleeping {delay:.1f}s",
                  flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")  # appease type checker

from context_harness.context_manager import ContextRole, ContextWindow
from context_harness.rag_pipeline import RAGPipeline, ChunkingStrategy
from evals.corpus import all_docs


MODEL = "gemini-2.5-flash"
MAX_TOKENS = 4096
MAX_TURNS = 8  # safety cap on the agent loop


SYSTEM_PROMPT = (
    "You are a Harry Potter lore expert. Answer questions using ONLY the "
    "information returned by the search_lore tool. If search_lore does not "
    "contain the answer, reply exactly: \"I don't know based on the available "
    "lore.\" Do not rely on outside knowledge.\n\n"
    "Always call search_lore at least once before answering. You may call it "
    "multiple times with different queries if the first result is insufficient."
)


# Gemini function-declaration schema (OpenAPI-ish JSON Schema subset)
SEARCH_LORE_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_lore",
            description=(
                "Search the Harry Potter lore corpus for passages relevant to a "
                "query. Returns up to top_k passages, each with a doc_id and text. "
                "Use this before answering any factual question."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Natural-language search query.",
                    ),
                    "top_k": types.Schema(
                        type=types.Type.INTEGER,
                        description="Max passages to return (default 3).",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)


@dataclass
class AgentRunResult:
    answer: str
    turns: int                    # number of model API calls
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int        # Gemini reports implicit caching separately
    cache_creation_tokens: int    # kept for API compat with Claude version
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
        }


class LoreAgent:
    """A tiny Gemini + RAG agent.

    Fields:
      self.rag    — RAGPipeline (search backend)
      self.window — ContextWindow for token-budget bookkeeping
      self.client — genai.Client
    """

    def __init__(
        self,
        rag: Optional[RAGPipeline] = None,
        client: Optional["genai.Client"] = None,
        max_context_tokens: int = 8000,
    ) -> None:
        self.rag = rag or self._build_default_rag()
        self.client = client or genai.Client()  # picks up GOOGLE_API_KEY
        self.window = ContextWindow(
            max_tokens=max_context_tokens,
            reserved_output_tokens=MAX_TOKENS,
        )
        self.window.add_text(ContextRole.SYSTEM, SYSTEM_PROMPT, priority=10.0)

    @staticmethod
    def _build_default_rag() -> RAGPipeline:
        rag = RAGPipeline(
            collection_name="hp_lore_agent",
            chunking_strategy=ChunkingStrategy.PARAGRAPH,
            top_k=3,
            use_chromadb=False,  # keyword fallback keeps things fast/offline
        )
        rag.ingest_many(all_docs())
        return rag

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict (Gemini function responses carry structured content)."""
        if name != "search_lore":
            return {"error": f"unknown tool: {name}"}
        query = inputs.get("query", "")
        top_k = int(inputs.get("top_k", 3))
        chunks = self.rag.retrieve(query, top_k=top_k)
        return {
            "passages": [
                {"doc_id": c.doc_id, "text": c.text, "score": round(c.score, 3)}
                for c in chunks
            ]
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def ask(self, question: str, verbose: bool = False) -> AgentRunResult:
        # Gemini's conversation is a list of Content objects with roles
        # "user" and "model" (not "assistant"). Tool calls come back as
        # FunctionCall parts on a model turn; tool results go back as
        # FunctionResponse parts on a user turn.
        contents: List[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=question)])
        ]

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[SEARCH_LORE_TOOL],
            max_output_tokens=MAX_TOKENS,
            # Gemini 2.5 flash defaults to auto function-calling via the SDK
            # helper; we opt into manual mode so we can log every step.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        total_in = total_out = cache_read = cache_create = 0
        tool_calls = 0
        transcript: List[Dict[str, Any]] = [{"role": "user", "content": question}]

        for turn in range(1, MAX_TURNS + 1):
            response = call_with_retry(lambda: self.client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            ))

            # Accumulate usage
            usage = response.usage_metadata
            if usage is not None:
                total_in += usage.prompt_token_count or 0
                total_out += usage.candidates_token_count or 0
                # Gemini implicit cache hits land here when present
                cache_read += getattr(usage, "cached_content_token_count", 0) or 0

            # Collect parts
            if not response.candidates:
                return AgentRunResult(
                    answer="(no candidates returned)",
                    turns=turn, tool_calls=tool_calls,
                    input_tokens=total_in, output_tokens=total_out,
                    cache_read_tokens=cache_read, cache_creation_tokens=cache_create,
                    transcript=transcript,
                )

            candidate = response.candidates[0]
            parts = candidate.content.parts or []

            # Log transcript entry
            turn_parts: List[Dict[str, Any]] = []
            function_calls: List[types.FunctionCall] = []
            text_out: List[str] = []
            for p in parts:
                if getattr(p, "text", None):
                    turn_parts.append({"type": "text", "text": p.text})
                    text_out.append(p.text)
                elif getattr(p, "function_call", None):
                    fc = p.function_call
                    function_calls.append(fc)
                    turn_parts.append({
                        "type": "function_call",
                        "name": fc.name,
                        "args": dict(fc.args or {}),
                    })

            transcript.append({
                "turn": turn,
                "finish_reason": str(candidate.finish_reason),
                "parts": turn_parts,
                "usage": {
                    "in": (usage.prompt_token_count or 0) if usage else 0,
                    "out": (usage.candidates_token_count or 0) if usage else 0,
                },
            })

            if verbose:
                print(f"\n--- turn {turn} (finish={candidate.finish_reason}) ---")
                for tp in turn_parts:
                    if tp["type"] == "text":
                        print(f"TEXT: {tp['text']}")
                    else:
                        print(f"FUNCTION_CALL: {tp['name']}({tp['args']})")

            # If the model produced function calls, execute them and feed back
            if function_calls:
                # Append the model's turn verbatim (with function_call parts)
                contents.append(candidate.content)

                # Dispatch each call, collect function responses
                response_parts: List[types.Part] = []
                tool_result_log: List[Dict[str, Any]] = []
                for fc in function_calls:
                    tool_calls += 1
                    result = self._dispatch_tool(fc.name, dict(fc.args or {}))
                    response_parts.append(
                        types.Part.from_function_response(
                            name=fc.name, response=result
                        )
                    )
                    tool_result_log.append({
                        "name": fc.name,
                        "args": dict(fc.args or {}),
                        "result": result,
                    })
                    if verbose:
                        preview = json.dumps(result)[:200]
                        print(f"FUNCTION_RESPONSE ({fc.name}): {preview}...")

                # Record tool results in transcript so eval_agent can inspect them
                transcript.append({
                    "turn": turn,
                    "role": "tool_results",
                    "tool_results": tool_result_log,
                })

                contents.append(types.Content(role="user", parts=response_parts))
                continue

            # No function calls — this is the final answer
            answer = "".join(text_out).strip() or "(agent produced no text)"
            return AgentRunResult(
                answer=answer, turns=turn, tool_calls=tool_calls,
                input_tokens=total_in, output_tokens=total_out,
                cache_read_tokens=cache_read, cache_creation_tokens=cache_create,
                transcript=transcript,
            )

        # Hit MAX_TURNS
        return AgentRunResult(
            answer="(agent exceeded max turns without finishing)",
            turns=MAX_TURNS, tool_calls=tool_calls,
            input_tokens=total_in, output_tokens=total_out,
            cache_read_tokens=cache_read, cache_creation_tokens=cache_create,
            transcript=transcript,
        )


def main():
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: set GOOGLE_API_KEY (get a free one at https://aistudio.google.com/apikey)",
              file=sys.stderr)
        sys.exit(2)

    if len(sys.argv) < 2:
        print("Usage: python -m evals.agent '<question>'")
        sys.exit(2)

    question = " ".join(sys.argv[1:])
    agent = LoreAgent()
    result = agent.ask(question, verbose=True)

    print("\n========= FINAL ANSWER =========")
    print(result.answer)
    print("\n========= STATS =========")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

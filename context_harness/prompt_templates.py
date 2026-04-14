"""
Prompt Templates
================
Structured prompt engineering for the deepTutor lore agent.

Context Engineering principles:
  1. **Separation of concerns** – system prompt (persona/rules) vs. user prompt (task).
  2. **Context slots** – explicit {{CONTEXT}} placeholders make injection traceable.
  3. **Instruction hierarchy** – most important constraints stated first AND last.
  4. **Chain-of-thought steering** – "Think step by step" only where it helps.
  5. **Negative space** – explicit "do not" rules reduce hallucination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Base template
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """
    A prompt template with named {{SLOT}} placeholders.

    Usage:
        tpl = PromptTemplate("Hello, {{NAME}}! You asked: {{QUERY}}")
        filled = tpl.render(NAME="Hermione", QUERY="What is Polyjuice Potion?")
    """

    template: str
    defaults: Dict[str, str] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        values = {**self.defaults, **kwargs}
        result = self.template
        for key, value in values.items():
            result = result.replace(f"{{{{{key}}}}}", value)
        missing = re.findall(r"\{\{(\w+)\}\}", result)
        if missing:
            raise ValueError(f"Unfilled template slots: {missing}")
        return result

    def slots(self) -> list[str]:
        return re.findall(r"\{\{(\w+)\}\}", self.template)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

class SystemPrompt:
    """
    Pre-built system prompts for the HP lore agent.

    Context engineering insight: the system prompt establishes the "frame"
    that every subsequent message is interpreted through.  Getting this right
    is more impactful than any individual retrieval tweak.
    """

    LORE_EXPERT = PromptTemplate(
        template="""\
You are a meticulous Harry Potter lore expert and tutor.
Your knowledge comes exclusively from the provided context passages.

Rules:
1. Only use information present in the CONTEXT block below.
2. If the context does not contain enough information, say so explicitly.
3. When citing information, reference the source number [N] from the context.
4. Do not speculate or invent facts not supported by the context.
5. Answer in clear, engaging language appropriate for a learner (age {{AUDIENCE_AGE}}).

{{CONTEXT}}

Remember: only use the context above. Do not rely on prior training knowledge.""",
        defaults={"AUDIENCE_AGE": "all ages", "CONTEXT": ""},
    )

    SOCRATIC_TUTOR = PromptTemplate(
        template="""\
You are a Socratic tutor specialising in Harry Potter lore.
Instead of giving direct answers, guide the student to discover the answer themselves
by asking probing questions.

Context to draw from:
{{CONTEXT}}

Teaching style:
- Start with a clarifying question to probe the student's current understanding.
- Provide hints that point toward the answer without revealing it directly.
- Celebrate when the student reaches the correct conclusion.
- Age / level: {{AUDIENCE_AGE}}""",
        defaults={"AUDIENCE_AGE": "secondary school", "CONTEXT": ""},
    )

    QUIZ_MASTER = PromptTemplate(
        template="""\
You are a quiz master for a Harry Potter lore quiz.
Using ONLY the context below, generate {{NUM_QUESTIONS}} multiple-choice questions.

Context:
{{CONTEXT}}

Format each question as:
Q[N]: <question text>
A) <option>  B) <option>  C) <option>  D) <option>
Answer: <letter>
Explanation: <one sentence from context>

Do not include questions whose answers cannot be found in the context.""",
        defaults={"NUM_QUESTIONS": "3", "CONTEXT": ""},
    )


# ---------------------------------------------------------------------------
# User / turn prompts
# ---------------------------------------------------------------------------

class LorePrompt:
    """Structured user-turn prompts for the lore agent."""

    QUESTION = PromptTemplate(
        template="""\
Question: {{QUERY}}

Please answer using only the context provided in the system message.
If you cite a source, use the [N] notation."""
    )

    FOLLOW_UP = PromptTemplate(
        template="""\
Follow-up: {{QUERY}}

My previous understanding: {{PRIOR_ANSWER}}

Please clarify or expand, still using only the provided context."""
    )

    COMPARE = PromptTemplate(
        template="""\
Compare and contrast: {{ENTITY_A}} vs {{ENTITY_B}}

Use only the context provided. Structure your answer as:
- Similarities:
- Differences:
- Which is more significant in the story, and why:"""
    )

    TIMELINE = PromptTemplate(
        template="""\
Build a chronological timeline of events related to: {{TOPIC}}

Use only the context provided.  Format as a numbered list with approximate dates / book references."""
    )


# ---------------------------------------------------------------------------
# Context injection helper
# ---------------------------------------------------------------------------

def inject_context(template: PromptTemplate, context_block: str, **kwargs) -> str:
    """
    Convenience wrapper: inject a context block into a template.

    Wraps the context in XML-style tags so the LLM treats it as structured data,
    which has been shown to improve grounding in several studies.
    """
    wrapped = f"<context>\n{context_block}\n</context>" if context_block else ""
    return template.render(CONTEXT=wrapped, **kwargs)

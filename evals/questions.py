"""
Eval question set
=================
Each question carries:
  - ``question``: natural language query
  - ``expected_docs``: set of doc_ids that must appear in retrieval to count as a hit
  - ``kind``: easy | multi | distractor — so we can report metrics by question type
  - ``reference_answer``: short ground-truth answer for LLM-as-judge grading

'easy'       -> answer is in a single passage
'multi'      -> answer requires combining info from 2+ passages
'distractor' -> answer is NOT in the corpus; agent should say "I don't know"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_docs: Set[str]
    kind: str                  # "easy" | "multi" | "distractor"
    reference_answer: str


QUESTIONS: List[EvalQuestion] = [
    # ---- easy: single passage ----
    EvalQuestion(
        id="q1", kind="easy",
        question="Who killed Dumbledore?",
        expected_docs={"dumbledore", "snape"},
        reference_answer="Severus Snape killed Dumbledore atop the Astronomy Tower, by prior arrangement with Dumbledore himself.",
    ),
    EvalQuestion(
        id="q2", kind="easy",
        question="What are the three Deathly Hallows?",
        expected_docs={"deathly_hallows"},
        reference_answer="The Elder Wand, the Resurrection Stone, and the Cloak of Invisibility.",
    ),
    EvalQuestion(
        id="q3", kind="easy",
        question="What is the incantation of the Patronus Charm?",
        expected_docs={"patronus"},
        reference_answer="Expecto Patronum.",
    ),
    EvalQuestion(
        id="q4", kind="easy",
        question="How many Horcruxes did Voldemort create?",
        expected_docs={"horcrux", "voldemort"},
        reference_answer="Seven (including Harry Potter, whom he created unintentionally).",
    ),
    EvalQuestion(
        id="q5", kind="easy",
        question="Where is Diagon Alley located?",
        expected_docs={"diagon_alley"},
        reference_answer="In London, behind a magical brick wall at the back of the Leaky Cauldron.",
    ),
    EvalQuestion(
        id="q6", kind="easy",
        question="What form does Harry Potter's Patronus take?",
        expected_docs={"patronus"},
        reference_answer="A stag, matching his father James's Animagus form.",
    ),

    # ---- multi: requires combining passages ----
    EvalQuestion(
        id="q7", kind="multi",
        question="Why are Harry's and Voldemort's wands considered brother wands?",
        expected_docs={"harry", "wands"},
        reference_answer="Both wands contain phoenix feathers from the same phoenix, Dumbledore's phoenix Fawkes.",
    ),
    EvalQuestion(
        id="q8", kind="multi",
        question="Which dark wizard did Dumbledore defeat, and how did that relate to the Elder Wand?",
        expected_docs={"dumbledore", "deathly_hallows"},
        reference_answer="Dumbledore defeated Gellert Grindelwald in 1945 and won the Elder Wand from him; the Elder Wand is one of the three Deathly Hallows.",
    ),
    EvalQuestion(
        id="q9", kind="multi",
        question="What was Snape's real motivation for working against Voldemort?",
        expected_docs={"snape", "harry"},
        reference_answer="His lifelong love of Harry's mother, Lily Potter.",
    ),
    EvalQuestion(
        id="q10", kind="multi",
        question="Which house at Hogwarts was Harry sorted into, and what does it value?",
        expected_docs={"harry", "hogwarts_houses"},
        reference_answer="Gryffindor, which values bravery and chivalry.",
    ),

    # ---- distractor: NOT in the corpus ----
    EvalQuestion(
        id="q11", kind="distractor",
        question="Who is the Minister for Magic during the Battle of Hogwarts?",
        expected_docs=set(),
        reference_answer="The corpus does not contain this information.",
    ),
    EvalQuestion(
        id="q12", kind="distractor",
        question="What is the population of Hogsmeade village?",
        expected_docs=set(),
        reference_answer="The corpus does not contain this information.",
    ),
]


def by_kind(kind: str) -> List[EvalQuestion]:
    return [q for q in QUESTIONS if q.kind == kind]

"""
Eval question set
=================
Each question carries:
  - ``question``: natural language query
  - ``expected_docs``: set of doc_ids that must appear in retrieval to count as a hit
  - ``kind``: easy | multi | inference | distractor
  - ``reference_answer``: short ground-truth answer for LLM-as-judge grading

Kinds:
  'easy'       -> answer is in a single passage
  'multi'      -> answer requires combining info from 2+ passages
  'inference'  -> corpus has partial evidence; agent should reason from it
  'distractor' -> answer is NOT in the corpus; agent should say "I don't know"

Doc IDs in this file must match data/hp_lore.txt exactly:
  albus-dumbledore, harry-potter, hermione-granger, ron-weasley,
  lord-voldemort, severus-snape, hogwarts, horcruxes,
  deathly-hallows, order-of-the-phoenix
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_docs: Set[str]
    kind: str                  # "easy" | "multi" | "inference" | "distractor"
    reference_answer: str


QUESTIONS: List[EvalQuestion] = [
    # ---- easy: answer is in a single passage ----
    EvalQuestion(
        id="q1", kind="easy",
        question="Who killed Dumbledore?",
        expected_docs={"albus-dumbledore", "severus-snape"},
        reference_answer="Severus Snape killed Dumbledore atop the Astronomy Tower, by prior arrangement with Dumbledore himself.",
    ),
    EvalQuestion(
        id="q2", kind="easy",
        question="What are the three Deathly Hallows?",
        expected_docs={"deathly-hallows"},
        reference_answer="The Elder Wand, the Resurrection Stone, and the Cloak of Invisibility.",
    ),
    EvalQuestion(
        id="q3", kind="easy",
        question="How many Horcruxes did Voldemort create?",
        expected_docs={"horcruxes", "lord-voldemort"},
        reference_answer="Seven (six intentional plus Harry Potter as an unintentional seventh).",
    ),
    EvalQuestion(
        id="q4", kind="easy",
        question="What form does Harry Potter's Patronus take?",
        expected_docs={"harry-potter"},
        reference_answer="A stag, matching his father James's Animagus form.",
    ),
    EvalQuestion(
        id="q5", kind="easy",
        question="Who founded the Order of the Phoenix?",
        expected_docs={"order-of-the-phoenix", "albus-dumbledore"},
        reference_answer="Albus Dumbledore founded the Order of the Phoenix to oppose Voldemort.",
    ),
    EvalQuestion(
        id="q6", kind="easy",
        question="Where is Hogwarts located?",
        expected_docs={"hogwarts"},
        reference_answer="In the Scottish Highlands, hidden from Muggles by enchantments.",
    ),
    EvalQuestion(
        id="q7", kind="easy",
        question="How did Voldemort die?",
        expected_docs={"lord-voldemort", "harry-potter"},
        reference_answer="Voldemort was killed by his own rebounding Killing Curse during the Battle of Hogwarts, because the Elder Wand's allegiance was to Harry.",
    ),
    EvalQuestion(
        id="q8", kind="easy",
        question="What did Hermione do after the Second Wizarding War?",
        expected_docs={"hermione-granger"},
        reference_answer="She returned to Hogwarts to complete her education, then joined the Ministry, eventually becoming Minister for Magic.",
    ),

    # ---- multi: requires combining info from 2+ passages ----
    EvalQuestion(
        id="q9", kind="multi",
        question="What was Snape's real motivation for working against Voldemort?",
        expected_docs={"severus-snape"},
        reference_answer="His lifelong love of Lily Evans (later Lily Potter). He defected to Dumbledore's side after Voldemort targeted her.",
    ),
    EvalQuestion(
        id="q10", kind="multi",
        question="Which Hogwarts house was Harry sorted into, and what traits does it value?",
        expected_docs={"harry-potter", "hogwarts", "ron-weasley"},
        reference_answer="Gryffindor, which values bravery, courage, and chivalry.",
    ),
    EvalQuestion(
        id="q11", kind="multi",
        question="How did Harry become the master of the Elder Wand without ever wielding it?",
        expected_docs={"deathly-hallows", "harry-potter"},
        reference_answer="Harry disarmed Draco Malfoy, who had unknowingly won the wand's allegiance from Dumbledore. The wand's loyalty transferred to Harry through the disarming.",
    ),
    EvalQuestion(
        id="q12", kind="multi",
        question="Name all known destroyers of Horcruxes and which Horcrux each destroyed.",
        expected_docs={"horcruxes"},
        reference_answer="Harry (diary, with Basilisk fang), Dumbledore (ring, with Sword of Gryffindor), Ron (locket, with Sword), Hermione (cup, with Basilisk fang), Crabbe (diadem, with Fiendfyre), Neville (Nagini, with Sword), Voldemort himself (Harry, with Killing Curse).",
    ),

    # ---- inference: corpus has partial evidence, agent should reason ----
    EvalQuestion(
        id="q13", kind="inference",
        question="Why might Dumbledore have trusted Snape despite Snape having been a Death Eater?",
        expected_docs={"severus-snape", "albus-dumbledore"},
        reference_answer="Because Snape defected after Voldemort targeted Lily Potter, and his lifelong love for Lily was proof of his changed loyalties. Dumbledore knew Snape's love was genuine motivation.",
    ),
    EvalQuestion(
        id="q14", kind="inference",
        question="What made Harry uniquely qualified to defeat Voldemort compared to other wizards?",
        expected_docs={"harry-potter", "lord-voldemort", "deathly-hallows"},
        reference_answer="Harry survived the Killing Curse as an infant (becoming an unintentional Horcrux), mastered all three Deathly Hallows, and won the Elder Wand's allegiance — a combination no other wizard achieved.",
    ),
    EvalQuestion(
        id="q15", kind="inference",
        question="How did the Order of the Phoenix's communication method relate to Dumbledore's Army?",
        expected_docs={"order-of-the-phoenix"},
        reference_answer="Order members communicated using Patronus messengers, and this technique was taught to Dumbledore's Army members.",
    ),

    # ---- distractor: NOT in the corpus, agent should say "I don't know" ----
    EvalQuestion(
        id="q16", kind="distractor",
        question="What is the incantation of the Patronus Charm?",
        expected_docs=set(),
        reference_answer="The corpus does not contain this information.",
    ),
    EvalQuestion(
        id="q17", kind="distractor",
        question="Where is Diagon Alley located?",
        expected_docs=set(),
        reference_answer="The corpus does not contain this information.",
    ),
    EvalQuestion(
        id="q18", kind="distractor",
        question="Who is the Minister for Magic during the Battle of Hogwarts?",
        expected_docs=set(),
        reference_answer="The corpus does not contain this information.",
    ),
    EvalQuestion(
        id="q19", kind="distractor",
        question="What is the population of Hogsmeade village?",
        expected_docs=set(),
        reference_answer="The corpus does not contain this information.",
    ),
    EvalQuestion(
        id="q20", kind="distractor",
        question="Why are Harry's and Voldemort's wands considered brother wands?",
        expected_docs=set(),
        reference_answer="The corpus does not contain this information (no mention of phoenix feather cores or Fawkes).",
    ),
]


def by_kind(kind: str) -> List[EvalQuestion]:
    return [q for q in QUESTIONS if q.kind == kind]

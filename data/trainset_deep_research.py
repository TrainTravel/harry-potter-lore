"""
Training data — Deep Research mode
===================================
Hand-labelled examples drawn from the 10 HP lore documents in data/hp_lore.txt.

Each example specifies the question and the expected citation doc_ids.
The `answer` field is intentionally left for BootstrapFewShot to generate —
labelling only what's cheap and objective (citations) while letting the
optimizer discover the phrasing that produces them.

Valid citation doc_ids (from data/hp_lore.txt):
    albus-dumbledore, harry-potter, hermione-granger, ron-weasley,
    lord-voldemort, severus-snape, hogwarts, horcruxes,
    deathly-hallows, order-of-the-phoenix
"""

from __future__ import annotations
import dspy


# ---------------------------------------------------------------------------
# Training set (used by BootstrapFewShot.compile)
# ---------------------------------------------------------------------------

TRAINSET = [
    dspy.Example(
        question="Who created the Horcruxes and how many were there?",
        citations="lord-voldemort horcruxes",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="What are the three Deathly Hallows?",
        citations="deathly-hallows",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Who killed Albus Dumbledore and why?",
        citations="albus-dumbledore severus-snape",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Which Hogwarts house was Hermione Granger sorted into?",
        citations="hermione-granger",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Where is Hogwarts located?",
        citations="hogwarts",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Who founded the Order of the Phoenix?",
        citations="order-of-the-phoenix albus-dumbledore",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="What is Harry Potter's Patronus?",
        citations="harry-potter",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="How did Harry Potter survive the Killing Curse as an infant?",
        citations="harry-potter lord-voldemort",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Who were Severus Snape's parents?",
        citations="severus-snape",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="How many children does Ron Weasley have and what are their names?",
        citations="ron-weasley",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Which Horcrux did Ron Weasley destroy?",
        citations="horcruxes ron-weasley",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Where did the Order of the Phoenix have its headquarters?",
        citations="order-of-the-phoenix",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Who were the four founders of Hogwarts?",
        citations="hogwarts",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="What was inadvertently the seventh Horcrux?",
        citations="horcruxes harry-potter",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="What is the Elder Wand and who are its known masters?",
        citations="deathly-hallows albus-dumbledore harry-potter",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="What career did Hermione Granger eventually pursue in the Ministry?",
        citations="hermione-granger",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Who did Severus Snape love, and how did that shape his choices?",
        citations="severus-snape harry-potter",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Name two members of the original Order of the Phoenix who died in the wars.",
        citations="order-of-the-phoenix",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="What Horcrux was hidden in Hufflepuff's cup, and who destroyed it?",
        citations="horcruxes hermione-granger",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="What was Voldemort's birth name and who were his parents?",
        citations="lord-voldemort",
        confidence="high",
    ).with_inputs("question"),
]


# ---------------------------------------------------------------------------
# Held-out evaluation set — do NOT use in BootstrapFewShot.compile
# ---------------------------------------------------------------------------

EVALSET = [
    dspy.Example(
        question="Who first owned the Invisibility Cloak in the Peverell family?",
        citations="deathly-hallows",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="What Horcrux did Dumbledore destroy and with what weapon?",
        citations="horcruxes albus-dumbledore",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Who is the headmaster of Hogwarts and what wand did he wield?",
        citations="albus-dumbledore deathly-hallows",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="What is the Hogwarts Express and where does it depart from?",
        citations="hogwarts",
        confidence="high",
    ).with_inputs("question"),

    dspy.Example(
        question="Did Hermione Granger ever become Minister for Magic?",
        citations="hermione-granger",
        confidence="high",
    ).with_inputs("question"),
]

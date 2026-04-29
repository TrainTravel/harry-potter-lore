"""
Training data — Open Analysis mode
====================================
Literary, psychological, and ethical analysis questions about HP lore.

Unlike other modes, open_analysis explicitly allows the LLM to reason beyond
the corpus. The Signature says: "Do not refuse — if the corpus is thin, lean
on your general knowledge and say so." So examples here may probe topics that
are only partially covered by the 10 lore documents.

Output fields:
  - analysis:      blended corpus + broader knowledge interpretation (≥100 chars)
  - corpus_facts:  concrete facts sourced from retrieved context (≥20 chars)
  - own_reasoning: interpretations or analogies beyond the corpus (≥30 chars)
  - citations:     space-separated doc_ids grounding corpus_facts

Valid citation doc_ids:
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
        question="Why is Snape such an enduring and morally complex character?",
        citations="severus-snape albus-dumbledore harry-potter",
    ).with_inputs("question"),

    dspy.Example(
        question="What does Voldemort's obsession with immortality reveal about him psychologically?",
        citations="lord-voldemort horcruxes",
    ).with_inputs("question"),

    dspy.Example(
        question="Was Dumbledore's manipulation of Harry ethical? Analyse the power dynamic.",
        citations="albus-dumbledore harry-potter",
    ).with_inputs("question"),

    dspy.Example(
        question="What does the pureblood/Muggle-born class divide represent thematically in HP?",
        citations="hogwarts hermione-granger lord-voldemort",
    ).with_inputs("question"),

    dspy.Example(
        question="Analyse Hermione Granger's relationship with academic achievement and perfectionism.",
        citations="hermione-granger hogwarts",
    ).with_inputs("question"),

    dspy.Example(
        question="Does Ron Weasley's inferiority complex make him a more or less sympathetic character?",
        citations="ron-weasley harry-potter",
    ).with_inputs("question"),

    dspy.Example(
        question="What does Horcrux creation reveal about Voldemort's understanding of love and connection?",
        citations="horcruxes lord-voldemort",
    ).with_inputs("question"),

    dspy.Example(
        question="Is Harry Potter a hero or merely a 'chosen one'? Does the distinction matter?",
        citations="harry-potter albus-dumbledore deathly-hallows",
    ).with_inputs("question"),

    dspy.Example(
        question="Analyse the theme of chosen destiny versus free will throughout Harry's story.",
        citations="harry-potter albus-dumbledore horcruxes",
    ).with_inputs("question"),

    dspy.Example(
        question="What does the Elder Wand's violent ownership history say about the corrupting nature of power?",
        citations="deathly-hallows lord-voldemort albus-dumbledore",
    ).with_inputs("question"),

    dspy.Example(
        question="How does Hagrid's marginalisation within wizarding society reflect broader themes of prejudice?",
        citations="hogwarts",
    ).with_inputs("question"),

    dspy.Example(
        question="What does Neville Longbottom's arc suggest about latent potential versus early performance?",
        citations="hogwarts order-of-the-phoenix",
    ).with_inputs("question"),
]


# ---------------------------------------------------------------------------
# Held-out evaluation set — do NOT use in BootstrapFewShot.compile
# ---------------------------------------------------------------------------

EVALSET = [
    dspy.Example(
        question="What does Draco Malfoy's arc reveal about peer pressure, inherited prejudice, and moral choice?",
        citations="hogwarts lord-voldemort",
    ).with_inputs("question"),

    dspy.Example(
        question="Analyse the Order of the Phoenix as a study in institutional resistance and the cost of dissent.",
        citations="order-of-the-phoenix albus-dumbledore severus-snape",
    ).with_inputs("question"),

    dspy.Example(
        question="How does J.K. Rowling use the Deathly Hallows to argue that accepting death is wiser than conquering it?",
        citations="deathly-hallows lord-voldemort harry-potter",
    ).with_inputs("question"),

    dspy.Example(
        question="Analyse McGonagall as a character study in duty, institutional loyalty, and the limits of rule-following.",
        citations="hogwarts order-of-the-phoenix",
    ).with_inputs("question"),

    dspy.Example(
        question="What does Sirius Black's tragedy reveal about the long-term psychological costs of unjust imprisonment?",
        citations="order-of-the-phoenix harry-potter",
    ).with_inputs("question"),

    dspy.Example(
        question="How does the HP series use Voldemort's inability to love as both a plot device and a moral argument?",
        citations="lord-voldemort horcruxes harry-potter albus-dumbledore",
    ).with_inputs("question"),
]

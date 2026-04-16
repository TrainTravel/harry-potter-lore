"""
Training data — Debate mode
============================
Hand-labelled examples for debatable positions in HP canon.

Each example specifies:
  - position: the claim being debated (passed as `question` by DSPyAgent.forward)
  - citations: doc_ids that good arguments MUST reference

The `arguments_for`, `arguments_against`, and `verdict` fields are left for
BootstrapFewShot to generate — we only label what is cheap and objective.

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
        position="Severus Snape was ultimately a hero, not a villain",
        citations="severus-snape harry-potter albus-dumbledore",
    ).with_inputs("position"),

    dspy.Example(
        position="Dumbledore manipulated Harry for the greater good at Harry's expense",
        citations="albus-dumbledore harry-potter horcruxes",
    ).with_inputs("position"),

    dspy.Example(
        position="Voldemort's defeat was inevitable because of his inability to understand love",
        citations="lord-voldemort harry-potter horcruxes",
    ).with_inputs("position"),

    dspy.Example(
        position="The Deathly Hallows were more dangerous than the Horcruxes",
        citations="deathly-hallows horcruxes",
    ).with_inputs("position"),

    dspy.Example(
        position="Hermione Granger was the most capable of the trio",
        citations="hermione-granger harry-potter ron-weasley",
    ).with_inputs("position"),

    dspy.Example(
        position="Hogwarts was a safe place for students during the Second Wizarding War",
        citations="hogwarts order-of-the-phoenix lord-voldemort",
    ).with_inputs("position"),

    dspy.Example(
        position="The Order of the Phoenix was effective in combating Voldemort",
        citations="order-of-the-phoenix albus-dumbledore lord-voldemort",
    ).with_inputs("position"),

    dspy.Example(
        position="Ron Weasley was undervalued compared to his contributions",
        citations="ron-weasley harry-potter hermione-granger",
    ).with_inputs("position"),

    dspy.Example(
        position="Destroying the Horcruxes was more important than the Deathly Hallows in defeating Voldemort",
        citations="horcruxes deathly-hallows harry-potter lord-voldemort",
    ).with_inputs("position"),

    dspy.Example(
        position="Snape's love for Lily Potter was his defining motivation above all else",
        citations="severus-snape harry-potter",
    ).with_inputs("position"),

    dspy.Example(
        position="Voldemort could have been defeated earlier if the Order of the Phoenix had acted differently",
        citations="order-of-the-phoenix lord-voldemort albus-dumbledore",
    ).with_inputs("position"),

    dspy.Example(
        position="Harry Potter succeeded because of luck rather than skill",
        citations="harry-potter lord-voldemort horcruxes deathly-hallows",
    ).with_inputs("position"),

    dspy.Example(
        position="Hogwarts' four-house system creates harmful division among students",
        citations="hogwarts",
    ).with_inputs("position"),

    dspy.Example(
        position="Dumbledore should have told Harry the full truth earlier",
        citations="albus-dumbledore harry-potter horcruxes",
    ).with_inputs("position"),

    dspy.Example(
        position="The Elder Wand brought more harm than good to all its owners",
        citations="deathly-hallows albus-dumbledore harry-potter lord-voldemort severus-snape",
    ).with_inputs("position"),
]


# ---------------------------------------------------------------------------
# Held-out evaluation set — do NOT use in BootstrapFewShot.compile
# ---------------------------------------------------------------------------

EVALSET = [
    dspy.Example(
        position="Hermione Granger would have made a better leader than Harry Potter",
        citations="hermione-granger harry-potter",
    ).with_inputs("position"),

    dspy.Example(
        position="The Horcrux strategy made Voldemort weaker, not stronger",
        citations="horcruxes lord-voldemort",
    ).with_inputs("position"),

    dspy.Example(
        position="Snape's double-agent role was the decisive factor in Voldemort's defeat",
        citations="severus-snape albus-dumbledore lord-voldemort",
    ).with_inputs("position"),

    dspy.Example(
        position="The Deathly Hallows legend was more myth than reality",
        citations="deathly-hallows",
    ).with_inputs("position"),

    dspy.Example(
        position="Ron Weasley's abandonment during the Horcrux hunt disqualifies him as a loyal friend",
        citations="ron-weasley horcruxes",
    ).with_inputs("position"),
]

"""
Training data — Satirical Podcast mode
========================================
Each example provides a `topic` (the HP magical thing being satirised) and
a `modern_angle` (the mundane contemporary lens applied for comedic effect).

`topic` is passed as `question` by DSPyAgent.forward; the module maps it
to the `topic` signature field and accepts `modern_angle` as a kwarg.

We do NOT label transcript/comedic_tension — BootstrapFewShot generates
those. We only label `citations` so the metric can check canon grounding.

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
        topic="invisibility cloak delivery via owl post",
        modern_angle="online marketplace and multi-day shipping frustrations",
        citations="deathly-hallows",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="love potions and obsession magic",
        modern_angle="dating app ethics and consent culture",
        citations="hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Sorting Hat house assignment",
        modern_angle="career aptitude tests and LinkedIn personal branding",
        citations="hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="house elves doing unpaid magical labour",
        modern_angle="gig economy worker rights and Deliveroo",
        citations="hogwarts hermione-granger",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Horcruxes as immortality strategy",
        modern_angle="cloud backup services and digital legacy planning",
        citations="horcruxes lord-voldemort",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Quidditch as a professional sport",
        modern_angle="athlete influencer culture and brand sponsorships",
        citations="hogwarts harry-potter",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Divination class and reading tea leaves",
        modern_angle="astrology apps and daily horoscope influencers",
        citations="hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Snape's potions class grading methods",
        modern_angle="toxic workplace culture and HR complaints",
        citations="severus-snape hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Dumbledore keeping crucial secrets from Harry",
        modern_angle="corporate management withholding information from employees",
        citations="albus-dumbledore harry-potter",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="the Elder Wand changing hands through violence",
        modern_angle="startup acquisition culture and hostile takeovers",
        citations="deathly-hallows",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Voldemort's no-nose appearance after Horcrux creation",
        modern_angle="extreme cosmetic surgery and body image influencers",
        citations="lord-voldemort horcruxes",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Hermione using a Time-Turner to attend extra classes",
        modern_angle="hustle culture and productivity influencers",
        citations="hermione-granger hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Order of the Phoenix operating as a secret resistance group",
        modern_angle="private group chats and encrypted messaging apps",
        citations="order-of-the-phoenix albus-dumbledore",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Ron Weasley's hand-me-down robes and second-hand school supplies",
        modern_angle="fast fashion, thrifting culture and class anxiety on social media",
        citations="ron-weasley hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Azkaban prison run by Dementors",
        modern_angle="private prison industry and corporate-run mental health facilities",
        citations="order-of-the-phoenix",
    ).with_inputs("topic", "modern_angle"),

    # --- new batch: radio show / fan mail / internet culture ---

    dspy.Example(
        topic="healing potions brewed in Snape's dungeon for everyday ailments",
        modern_angle="pharmaceutical industry direct-to-consumer ads and prescription drug pricing",
        citations="severus-snape hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="owl post delivery tracking and data plan bureaucracy",
        modern_angle="mobile carrier hidden fees and 'unlimited' plan small print",
        citations="hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Taylor Swift performing a concert at Hogwarts",
        modern_angle="concert ticket scalping, Eras Tour chaos and celebrity stan culture",
        citations="hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="daily life inside Azkaban under Dementor supervision",
        modern_angle="reality TV prison documentaries and inmate influencer accounts",
        citations="order-of-the-phoenix harry-potter",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Engorgio and shrinking charms used for cosmetic body modification",
        modern_angle="plastic surgery culture, BBL trends and before-and-after content creators",
        citations="hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="a magical radio show answering listener fan mail",
        modern_angle="podcast Q&A culture, parasocial relationships and Patreon perks",
        citations="hogwarts order-of-the-phoenix",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="dark magic addiction and forced rehabilitation after the wizarding war",
        modern_angle="wellness retreats, trauma therapy discourse and recovery influencers",
        citations="lord-voldemort order-of-the-phoenix",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="wizard memes and magical internet culture on the wizarding web",
        modern_angle="social media algorithms, viral content and chronically online humour",
        citations="hogwarts",
    ).with_inputs("topic", "modern_angle"),
]


# ---------------------------------------------------------------------------
# Held-out evaluation set — do NOT use in BootstrapFewShot.compile
# ---------------------------------------------------------------------------

EVALSET = [
    dspy.Example(
        topic="Hogwarts acceptance letters delivered by owls",
        modern_angle="university admissions marketing and spam email",
        citations="hogwarts",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Voldemort's return via Horcrux resurrection ritual",
        modern_angle="tech founder comeback narratives and 'return from exile' PR",
        citations="lord-voldemort horcruxes",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Snape's double-agent secret identity",
        modern_angle="social media personas and living a double life online",
        citations="severus-snape albus-dumbledore",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="the Deathly Hallows as the ultimate power trio",
        modern_angle="collector culture and limited edition product drops",
        citations="deathly-hallows",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Harry Potter being famous from birth for surviving a curse",
        modern_angle="nepotism babies and unearned celebrity culture",
        citations="harry-potter lord-voldemort",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Azkaban inmates writing letters home after being released",
        modern_angle="post-incarceration reintegration and LinkedIn job hunting",
        citations="order-of-the-phoenix",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="magical radio DJs taking live calls from wizard listeners",
        modern_angle="late-night talk radio parasocial intimacy and caller culture",
        citations="hogwarts order-of-the-phoenix",
    ).with_inputs("topic", "modern_angle"),

    dspy.Example(
        topic="Madam Pomfrey's hospital wing versus muggle A&E",
        modern_angle="NHS waiting times and private healthcare price gouging",
        citations="hogwarts",
    ).with_inputs("topic", "modern_angle"),
]

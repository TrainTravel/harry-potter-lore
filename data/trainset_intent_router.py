"""
Training data — Intent Router (mode=auto)
==========================================
Hand-labelled examples for classifying user messages into agent modes.

Each example specifies:
  - user_message: the raw user input
  - mode: the correct mode to route to
  - kwargs_json: JSON string with extracted parameters
  - confidence: expected confidence level

Canonical rule: ``.with_inputs("user_message")`` matches the
IntentRouterSignature's input field name verbatim.
"""

from __future__ import annotations
import dspy


# ---------------------------------------------------------------------------
# Training set (used by BootstrapFewShot.compile)
# ---------------------------------------------------------------------------

TRAINSET = [
    # --- deep_research (6 examples) ---
    dspy.Example(
        user_message="What are the properties of the Elder Wand?",
        mode="deep_research",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Tell me about the Battle of Hogwarts",
        mode="deep_research",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="What spells did Voldemort use?",
        mode="deep_research",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Who were the original members of the Order of the Phoenix?",
        mode="deep_research",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="How does the Marauder's Map work?",
        mode="deep_research",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="List all the Horcruxes and where they were hidden",
        mode="deep_research",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # --- guided_learning (5 examples) ---
    dspy.Example(
        user_message="Help me understand how Horcruxes work",
        mode="guided_learning",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="I'm trying to learn about the Patronus charm, can you teach me?",
        mode="guided_learning",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Quiz me on the Deathly Hallows",
        mode="guided_learning",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Can you walk me through Dumbledore's plan step by step?",
        mode="guided_learning",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="I want to study the history of Hogwarts houses",
        mode="guided_learning",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # --- exam_grader (5 examples) ---
    dspy.Example(
        user_message="Grade this: Q: What are the Deathly Hallows? A: Three powerful objects created by Death himself - the Elder Wand, the Resurrection Stone, and the Cloak of Invisibility.",
        mode="exam_grader",
        kwargs_json='{"student_answer": "Three powerful objects created by Death himself - the Elder Wand, the Resurrection Stone, and the Cloak of Invisibility."}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Check my answer: The question was about who created Horcruxes. My answer: Voldemort invented Horcruxes to become immortal.",
        mode="exam_grader",
        kwargs_json='{"student_answer": "Voldemort invented Horcruxes to become immortal."}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Is this correct? Snape was a Death Eater who never changed sides.",
        mode="exam_grader",
        kwargs_json='{"student_answer": "Snape was a Death Eater who never changed sides."}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Grade my essay on the Triwizard Tournament: The tournament has four tasks and is held every year at Hogwarts.",
        mode="exam_grader",
        kwargs_json='{"student_answer": "The tournament has four tasks and is held every year at Hogwarts."}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Mark this answer: Q: Name three Unforgivable Curses. A: Avada Kedavra, Crucio, and Expelliarmus.",
        mode="exam_grader",
        kwargs_json='{"student_answer": "Avada Kedavra, Crucio, and Expelliarmus."}',
        confidence="high",
    ).with_inputs("user_message"),

    # --- open_analysis (5 examples) ---
    dspy.Example(
        user_message="Why did Snape become the person he became?",
        mode="open_analysis",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="What does the Mirror of Erised say about human psychology?",
        mode="open_analysis",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Compare the leadership styles of Dumbledore and Voldemort",
        mode="open_analysis",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Is Hogwarts an effective educational institution by modern standards?",
        mode="open_analysis",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Analyze the class system in wizarding society",
        mode="open_analysis",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # --- perspective_shift (6 examples) ---
    dspy.Example(
        user_message="What would Dumbledore say about my career change?",
        mode="perspective_shift",
        kwargs_json='{"character": "Dumbledore"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Sort me into a house",
        mode="perspective_shift",
        kwargs_json='{"character": "Sorting Hat"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Put me into a house",
        mode="perspective_shift",
        kwargs_json='{"character": "Sorting Hat"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="I'm going through a hard time. What would Luna say?",
        mode="perspective_shift",
        kwargs_json='{"character": "Luna"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="How would Hermione handle imposter syndrome at a new job?",
        mode="perspective_shift",
        kwargs_json='{"character": "Hermione"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Give me a pep talk as McGonagall",
        mode="perspective_shift",
        kwargs_json='{"character": "McGonagall"}',
        confidence="high",
    ).with_inputs("user_message"),

    # Character-less life advice. The user describes a personal situation
    # without naming an HP character. The router must still route to
    # perspective_shift (kwargs empty); the API short-circuits with
    # character options so the user picks. Without these examples the
    # router classifies bare life advice as "none" (off-topic) because
    # it has no HP keyword to anchor on.
    dspy.Example(
        user_message="i love someone but they don't actively reply to my messages",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="How do I deal with imposter syndrome at a new job?",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="My best friend and I are drifting apart — what do I do?",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="I'm afraid of disappointing my family with my career choices",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="I'm stuck between a safe job and a creative path I love",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="I keep procrastinating on the work that actually matters",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # Grief / loss / mental-health — heavy real-life situations the
    # product is explicitly built to engage with via character voice.
    # Two examples is enough; the broader Signature description carries
    # the generalization.
    dspy.Example(
        user_message="What do you think about a mother who lost her sons?",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="I'm grieving the loss of my brother and don't know how to keep going",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # --- debate (5 examples) ---
    dspy.Example(
        user_message="Was Snape really a hero?",
        mode="debate",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Argue both sides: were the Deathly Hallows more dangerous than Horcruxes?",
        mode="debate",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Debate whether Dumbledore was right to keep secrets from Harry",
        mode="debate",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Make the case for and against the Statute of Secrecy",
        mode="debate",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Is Voldemort's defeat inevitable or could he have won?",
        mode="debate",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # --- satirical_podcast (5 examples) ---
    dspy.Example(
        user_message="Do a podcast about Quidditch as an extreme sport industry",
        mode="satirical_podcast",
        kwargs_json='{"modern_angle": "extreme sport industry"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Make a funny podcast about house elves and gig economy labor",
        mode="satirical_podcast",
        kwargs_json='{"modern_angle": "gig economy labor"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Satirize the Daily Prophet as a modern clickbait news outlet",
        mode="satirical_podcast",
        kwargs_json='{"modern_angle": "clickbait news outlet"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Do a comedy podcast episode about Hogwarts as a tech startup",
        mode="satirical_podcast",
        kwargs_json='{"modern_angle": "tech startup"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Podcast: wizarding dating apps",
        mode="satirical_podcast",
        kwargs_json='{"modern_angle": "dating apps"}',
        confidence="high",
    ).with_inputs("user_message"),

    # --- none (off-topic, 5 examples) ---
    dspy.Example(
        user_message="What's the weather like today?",
        mode="none",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Solve this equation: 2x + 3 = 7",
        mode="none",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Tell me about the Lord of the Rings",
        mode="none",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="What's the capital of France?",
        mode="none",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Write me a Python function to sort a list",
        mode="none",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # --- boundary/ambiguous (3 examples) ---
    dspy.Example(
        user_message="Tell me about Snape",
        mode="deep_research",
        kwargs_json="{}",
        confidence="medium",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="What do you think about Dumbledore?",
        mode="open_analysis",
        kwargs_json="{}",
        confidence="medium",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Snape",
        mode="deep_research",
        kwargs_json="{}",
        confidence="low",
    ).with_inputs("user_message"),
]


# ---------------------------------------------------------------------------
# Held-out evaluation set
# ---------------------------------------------------------------------------

EVALSET = [
    # deep_research
    dspy.Example(
        user_message="What is the Fidelius Charm and how does it work?",
        mode="deep_research",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Name all the known Animagi in the Harry Potter series",
        mode="deep_research",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # guided_learning
    dspy.Example(
        user_message="Teach me about the Unforgivable Curses",
        mode="guided_learning",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="I want to learn about wand lore, can you guide me?",
        mode="guided_learning",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # exam_grader
    dspy.Example(
        user_message="Grade this: Q: Who is the Half-Blood Prince? A: It's Harry Potter because he's half wizard half muggle.",
        mode="exam_grader",
        kwargs_json='{"student_answer": "It\'s Harry Potter because he\'s half wizard half muggle."}',
        confidence="high",
    ).with_inputs("user_message"),

    # open_analysis
    dspy.Example(
        user_message="What are the ethical implications of memory charms in HP?",
        mode="open_analysis",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # perspective_shift
    dspy.Example(
        user_message="What house would I be in?",
        mode="perspective_shift",
        kwargs_json='{"character": "Sorting Hat"}',
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="How would Snape handle a difficult coworker?",
        mode="perspective_shift",
        kwargs_json='{"character": "Snape"}',
        confidence="high",
    ).with_inputs("user_message"),

    # debate
    dspy.Example(
        user_message="Did the Ministry of Magic do more harm than good?",
        mode="debate",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # satirical_podcast
    dspy.Example(
        user_message="Make a comedy bit about Owl Post vs email",
        mode="satirical_podcast",
        kwargs_json='{"modern_angle": "email vs traditional mail"}',
        confidence="high",
    ).with_inputs("user_message"),

    # none
    dspy.Example(
        user_message="How do I cook pasta?",
        mode="none",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="What is quantum computing?",
        mode="none",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    # boundary
    dspy.Example(
        user_message="Hogwarts",
        mode="deep_research",
        kwargs_json="{}",
        confidence="low",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Is magic real?",
        mode="open_analysis",
        kwargs_json="{}",
        confidence="medium",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="Talk to me as if you were Hagrid dealing with a layoff",
        mode="perspective_shift",
        kwargs_json='{"character": "Hagrid"}',
        confidence="high",
    ).with_inputs("user_message"),

    # Character-less life advice (held-out)
    dspy.Example(
        user_message="I can't get over a breakup and I don't know what to do",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),

    dspy.Example(
        user_message="How do I forgive a friend who hurt me badly?",
        mode="perspective_shift",
        kwargs_json="{}",
        confidence="high",
    ).with_inputs("user_message"),
]

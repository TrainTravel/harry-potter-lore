"""
Training data — Exam Grader mode
=================================
Hand-labelled examples for grading student answers against HP lore.

Each example provides the question, a student answer (varying quality),
and the expected score + passing status. The critique field is left for
BootstrapFewShot to generate — we only label what's cheap and objective.

Scoring rubric:
  90-100: fully correct, well-supported by corpus
  70-89:  mostly correct, minor omissions
  60-69:  passing but weak, missing key details
  40-59:  partially correct, significant errors
  0-39:   wrong or fabricated
"""

from __future__ import annotations
import dspy


TRAINSET = [
    # --- High-quality answers (should score 85+) ---
    dspy.Example(
        question="Who created the Horcruxes and how many were there?",
        student_answer="Lord Voldemort created seven Horcruxes by splitting his soul through murder. The seven pieces were: his diary, Gaunt's ring, Slytherin's locket, Hufflepuff's cup, Ravenclaw's diadem, Nagini, and Harry Potter (unintentionally).",
        expected_score=95,
        expected_passing=True,
    ).with_inputs("question", "student_answer"),

    dspy.Example(
        question="What are the three Deathly Hallows?",
        student_answer="The three Deathly Hallows are the Elder Wand, the Resurrection Stone, and the Invisibility Cloak. They were said to be created by Death and given to the three Peverell brothers.",
        expected_score=95,
        expected_passing=True,
    ).with_inputs("question", "student_answer"),

    dspy.Example(
        question="Who killed Dumbledore and why?",
        student_answer="Severus Snape killed Dumbledore at the top of the Astronomy Tower. It was part of Dumbledore's own plan to protect Draco Malfoy and maintain Snape's cover with Voldemort.",
        expected_score=90,
        expected_passing=True,
    ).with_inputs("question", "student_answer"),

    # --- Partially correct answers (should score 40-69) ---
    dspy.Example(
        question="Who created the Horcruxes and how many were there?",
        student_answer="Voldemort made some Horcruxes. I think there were six, including his diary and a snake.",
        expected_score=45,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    dspy.Example(
        question="What are the three Deathly Hallows?",
        student_answer="The Deathly Hallows are the Elder Wand, the Invisibility Cloak, and a magic stone. They make you immortal.",
        expected_score=55,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    dspy.Example(
        question="Where is Hogwarts located?",
        student_answer="Hogwarts is somewhere in England. It's a big castle.",
        expected_score=35,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    # --- Wrong answers (should score 0-30) ---
    dspy.Example(
        question="Who killed Dumbledore and why?",
        student_answer="Harry Potter killed Dumbledore because Dumbledore was secretly evil.",
        expected_score=5,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    dspy.Example(
        question="Who founded the Order of the Phoenix?",
        student_answer="The Order of the Phoenix was founded by Harry Potter in his fifth year at Hogwarts.",
        expected_score=10,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    # --- Correct but incomplete (should score 60-75) ---
    dspy.Example(
        question="What is a Horcrux?",
        student_answer="A Horcrux is an object containing a piece of someone's soul. You have to kill someone to make one.",
        expected_score=65,
        expected_passing=True,
    ).with_inputs("question", "student_answer"),

    dspy.Example(
        question="Who is Hermione Granger?",
        student_answer="Hermione is a witch who went to Hogwarts and was friends with Harry and Ron. She was very smart.",
        expected_score=60,
        expected_passing=True,
    ).with_inputs("question", "student_answer"),

    # --- Fabrication (should score very low) ---
    dspy.Example(
        question="What house was Ron Weasley sorted into?",
        student_answer="Ron Weasley was sorted into Slytherin, where he became friends with Draco Malfoy.",
        expected_score=0,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    dspy.Example(
        question="How did Harry defeat Voldemort?",
        student_answer="Harry defeated Voldemort by using a time-turner to go back and prevent Voldemort from ever being born.",
        expected_score=0,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),
]


# ---------------------------------------------------------------------------
# Held-out evaluation set — do NOT use in BootstrapFewShot.compile
# ---------------------------------------------------------------------------

EVALSET = [
    # Excellent — full explanation with mechanism
    dspy.Example(
        question="How did Lily Potter's sacrifice protect Harry from Voldemort's killing curse?",
        student_answer="Lily Potter chose to die rather than step aside, and that act of willing self-sacrifice created an ancient protective enchantment rooted in love. When Voldemort cast Avada Kedavra, the curse rebounded because of this protection. The magic lived in Harry's blood, which is why Voldemort's use of Harry's blood in his resurrection ritual paradoxically kept that protection alive.",
        expected_score=93,
        expected_passing=True,
    ).with_inputs("question", "student_answer"),

    # Good — correct but misses the blood-resurrection detail
    dspy.Example(
        question="What was Snape's true allegiance throughout the series?",
        student_answer="Snape was secretly loyal to Dumbledore, acting as a double agent inside Voldemort's Death Eaters. His underlying motivation was his love for Lily Potter, Harry's mother. He protected Harry throughout his time at Hogwarts and revealed crucial information about Horcruxes before he died.",
        expected_score=85,
        expected_passing=True,
    ).with_inputs("question", "student_answer"),

    # Passing but weak — correct facts, no depth
    dspy.Example(
        question="Who is Ron Weasley and what is his background?",
        student_answer="Ron Weasley is Harry Potter's best friend. He comes from a large wizarding family that doesn't have much money. He's very loyal and brave.",
        expected_score=62,
        expected_passing=True,
    ).with_inputs("question", "student_answer"),

    # Partially correct — right topic, misses half the Horcruxes
    dspy.Example(
        question="Name all of Voldemort's Horcruxes.",
        student_answer="Voldemort made his diary, Marvolo Gaunt's ring, Slytherin's locket, and Hufflepuff's cup into Horcruxes. I think there were a couple more but I can't remember them.",
        expected_score=48,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    # Partially correct — right topic but skips founding context
    dspy.Example(
        question="Describe the founding and purpose of the Order of the Phoenix.",
        student_answer="The Order of the Phoenix was a secret organisation that fought against Voldemort. It included wizards like Mad-Eye Moody and Lupin.",
        expected_score=50,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    # Wrong — fabricated founder
    dspy.Example(
        question="Who founded the Order of the Phoenix?",
        student_answer="The Order of the Phoenix was founded by Harry Potter and Hermione Granger in their sixth year at Hogwarts to fight Voldemort's Death Eaters.",
        expected_score=5,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    # Wrong — reverses canon fact
    dspy.Example(
        question="What is Hermione Granger known for academically?",
        student_answer="Hermione Granger was actually an average student who succeeded mainly through hard work and close friendship with Harry, not through any particular natural ability or intelligence.",
        expected_score=12,
        expected_passing=False,
    ).with_inputs("question", "student_answer"),

    # Correct but under-detailed (Deathly Hallows summary)
    dspy.Example(
        question="What are the Deathly Hallows and who created them according to legend?",
        student_answer="The Deathly Hallows are three magical objects: a wand, a stone, and a cloak. According to the legend, Death himself gave them to three brothers.",
        expected_score=68,
        expected_passing=True,
    ).with_inputs("question", "student_answer"),
]

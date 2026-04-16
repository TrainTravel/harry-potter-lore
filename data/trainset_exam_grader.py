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

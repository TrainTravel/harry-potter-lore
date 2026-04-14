"""
HP Lore Seed Corpus
===================
Small hand-written corpus for evaluating the RAG pipeline and agent.
Each passage has a stable ``doc_id`` so eval questions can assert that
retrieval surfaces the right one.

Keep passages small and single-topic so chunking behavior is interpretable.
"""

from __future__ import annotations
from typing import List, Tuple

# (doc_id, text)
CORPUS: List[Tuple[str, str]] = [
    (
        "dumbledore",
        "Albus Percival Wulfric Brian Dumbledore was the Headmaster of Hogwarts "
        "School of Witchcraft and Wizardry. He defeated the dark wizard Gellert "
        "Grindelwald in a legendary duel in 1945. Dumbledore was widely considered "
        "the greatest wizard of his age. He owned the Elder Wand, one of the three "
        "Deathly Hallows, having won it from Grindelwald. He was eventually killed "
        "by Severus Snape atop the Astronomy Tower, by prior arrangement, in 1997."
    ),
    (
        "voldemort",
        "Lord Voldemort, born Tom Marvolo Riddle, was the most feared dark wizard "
        "of the twentieth century. He founded the Death Eaters and sought to purge "
        "the wizarding world of Muggle-born witches and wizards. To escape death he "
        "split his soul into seven Horcruxes. His body was first destroyed when his "
        "Killing Curse rebounded off the infant Harry Potter in 1981."
    ),
    (
        "harry",
        "Harry James Potter, known as The Boy Who Lived, survived Voldemort's "
        "Killing Curse as an infant because of his mother Lily's protective "
        "sacrifice. He was sorted into Gryffindor House at Hogwarts. His wand "
        "contains a phoenix feather from Fawkes, Dumbledore's phoenix; the same "
        "phoenix gave a feather to Voldemort's wand, making them brother wands."
    ),
    (
        "horcrux",
        "A Horcrux is a dark magical object in which a wizard has concealed a "
        "fragment of their soul in order to attain immortality. Creating a Horcrux "
        "requires committing murder, which rips the soul. Voldemort created seven: "
        "the diary, Marvolo Gaunt's ring, Slytherin's locket, Hufflepuff's cup, "
        "Ravenclaw's diadem, the snake Nagini, and unintentionally, Harry Potter."
    ),
    (
        "deathly_hallows",
        "The Deathly Hallows are three powerful magical objects: the Elder Wand, "
        "the unbeatable wand; the Resurrection Stone, which can summon shades of "
        "the dead; and the Cloak of Invisibility, which hides its wearer from "
        "death itself. According to legend, the wizard who possesses all three "
        "becomes the Master of Death."
    ),
    (
        "hogwarts_houses",
        "Hogwarts School is divided into four houses, each founded by one of the "
        "four great medieval wizards. Gryffindor values bravery and chivalry; "
        "Slytherin values ambition and cunning; Ravenclaw values wit and learning; "
        "Hufflepuff values loyalty and hard work. Students are sorted by the "
        "Sorting Hat during their first-year feast."
    ),
    (
        "snape",
        "Severus Snape was Potions Master, and later Defence Against the Dark "
        "Arts teacher, at Hogwarts. He was a double agent working for Dumbledore "
        "against Voldemort, motivated by his lifelong love of Lily Potter. He "
        "killed Dumbledore on Dumbledore's own orders to spare Draco Malfoy from "
        "committing murder. Snape was killed by Nagini in the Shrieking Shack."
    ),
    (
        "patronus",
        "The Patronus Charm is a defensive spell that produces a silver guardian "
        "— the Patronus — which repels Dementors. The incantation is 'Expecto "
        "Patronum', and it requires the caster to focus on a single, powerful "
        "happy memory. Harry Potter's Patronus takes the form of a stag, matching "
        "his father James's Animagus form."
    ),
    (
        "wands",
        "Wands in the wizarding world typically consist of a wooden shaft with a "
        "magical core. The three supreme cores used by wandmaker Garrick Ollivander "
        "are phoenix feather, dragon heartstring, and unicorn hair. A wand chooses "
        "the wizard, not the other way around, and allegiance can transfer if the "
        "wand is won from its previous owner."
    ),
    (
        "diagon_alley",
        "Diagon Alley is a cobblestoned wizarding shopping street located behind "
        "a magical brick wall at the back of the Leaky Cauldron in London. It is "
        "home to Gringotts Wizarding Bank (run by goblins), Ollivanders wand shop, "
        "Flourish and Blotts bookshop, and Florean Fortescue's Ice Cream Parlour. "
        "Hogwarts students buy their school supplies there before each term."
    ),
]


def all_docs() -> List[Tuple[str, str]]:
    """Return (text, doc_id) tuples in the shape ``RAGPipeline.ingest_many`` wants."""
    return [(text, doc_id) for doc_id, text in CORPUS]


def doc_ids() -> List[str]:
    return [doc_id for doc_id, _ in CORPUS]

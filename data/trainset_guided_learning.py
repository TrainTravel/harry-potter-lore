"""
Training data — Guided Learning mode
=====================================
Socratic examples. We only label positive demonstrations — hints that
guide without revealing the answer, next_questions that actively probe,
and explanations that discuss the concept without giving the mechanism.

The metric in context_harness/metrics.socratic_metric is what enforces
the "never reveal the answer directly" rule during optimization.
"""

from __future__ import annotations
import dspy


TRAINSET = [
    dspy.Example(
        question="What is a Horcrux?",
        past_attempts="none",
        hint=(
            "Think about what someone might do if they feared death more than anything. "
            "What if a wizard could hide a piece of themselves in an object?"
        ),
        next_question="What kind of act would be required to split a soul in the first place?",
        explanation=(
            "The concept connects immortality with an irreversible form of dark magic. "
            "It hinges on the idea that the soul can be divided — the object is secondary "
            "to the act that fragments it."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Why did Snape behave cruelly toward Harry?",
        past_attempts="Maybe Snape hated James Potter and took it out on Harry?",
        hint=(
            "You're on the right track about James — but consider who else Snape knew "
            "in school. Whose memory did he carry for decades?"
        ),
        next_question="What might someone do to stay close to a child whose mother they once loved?",
        explanation=(
            "The behaviour makes sense only when you consider loyalty and grief "
            "operating underneath hostility. The surface cruelty is a mask for a "
            "much older commitment."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Why did Voldemort fail to kill Harry as a baby?",
        past_attempts="Was it because of some kind of magical shield?",
        hint=(
            "Close — but think about who cast the shield, and what they had to do. "
            "Is there any form of magic older than spells?"
        ),
        next_question="What kind of sacrifice creates lasting protection in the wizarding world?",
        explanation=(
            "The event hinges on a form of magic older than wandwork. It's not a "
            "charm or a counter-curse — it's what someone chose to do, and why."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="What makes the Elder Wand so powerful?",
        past_attempts="none",
        hint=(
            "Consider how ownership of a wand is normally decided. What if that rule "
            "operated more strictly for one particular wand?"
        ),
        next_question="How might one become the master of the Elder Wand without ever holding it?",
        explanation=(
            "The wand's power is tied to its allegiance, which follows victory rather "
            "than purchase or inheritance. Mastery is less about possession and more "
            "about a specific kind of event."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Why can't Voldemort use the Elder Wand effectively against Harry?",
        past_attempts="The Elder Wand didn't really belong to Voldemort because he killed Snape who didn't actually own it?",
        hint=(
            "Exactly the right thread. Now trace it one step further — if Snape wasn't "
            "the true owner, who had actually won the wand's allegiance earlier?"
        ),
        next_question="When did the wand's allegiance shift, and what simple act caused it?",
        explanation=(
            "The chain of mastery passes through an ordinary, almost forgettable "
            "moment — not the dramatic killing Voldemort witnessed. This is why the "
            "wand turns against its wielder at the decisive moment."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="How is the Sorting Hat able to read minds?",
        past_attempts="none",
        hint=(
            "Think about where the Hat originated. An object made by a powerful witch "
            "or wizard can carry something of them — what might Godric Gryffindor have "
            "left inside it?"
        ),
        next_question="What would it mean for a founder's intentions and judgment to persist in an object?",
        explanation=(
            "Enchanted objects in the wizarding world can hold more than a single "
            "spell — they can carry aspects of their creators' minds. The Hat is an "
            "example of this rather than an exception."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Why did Dumbledore trust Snape?",
        past_attempts="Because Snape switched sides after Lily's death?",
        hint=(
            "You're right about the trigger — but what specifically convinced Dumbledore "
            "that Snape's change was permanent rather than convenient?"
        ),
        next_question="What did Dumbledore ask Snape to do that a bad actor would never agree to?",
        explanation=(
            "Trust in the Order was rarely built on past loyalty alone. Dumbledore "
            "relied on commitments that could only be upheld by someone who had "
            "genuinely changed — not by someone merely pretending to."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Why didn't Harry die in the Forbidden Forest?",
        past_attempts="I'm not sure — did the curse miss?",
        hint=(
            "The curse didn't miss. Think about what Voldemort was actually destroying "
            "when he cast it at Harry — and why Harry's own body could survive that."
        ),
        next_question="If something of Voldemort was hidden inside Harry, what would a Killing Curse target first?",
        explanation=(
            "The event is best understood as Voldemort destroying his own foreign "
            "fragment rather than Harry himself. Harry's survival turns on what he was "
            "carrying, not on the curse failing."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="What makes Hogwarts hidden from Muggles?",
        past_attempts="none",
        hint=(
            "Magical concealment rarely works by simply blocking sight. What might a "
            "Muggle see instead, if the castle can't simply be invisible?"
        ),
        next_question="What kind of enchantment would discourage a Muggle from investigating further?",
        explanation=(
            "Concealment in the wizarding world often layers substitution and "
            "suggestion rather than pure invisibility — the Muggle sees something, "
            "but something that prompts them to leave."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Why is a Patronus unique to each wizard?",
        past_attempts="Because it reflects personality?",
        hint=(
            "Personality is part of it, but not the whole. The Patronus charm pulls "
            "from a deeper well — what emotion does the caster have to hold on to?"
        ),
        next_question="What connection between two people might cause their Patronuses to mirror each other?",
        explanation=(
            "A Patronus emerges from a specific psychological state the caster must "
            "access. Its form therefore encodes more than identity — it can encode "
            "love, loss, and imprinting."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="How can a chess set sacrifice itself for Ron?",
        past_attempts="none",
        hint=(
            "Start with what Wizard's Chess already does on an ordinary board — the "
            "pieces aren't inert. How might McGonagall extend that to an obstacle course?"
        ),
        next_question="What transfiguration or charm might animate chess pieces at human scale?",
        explanation=(
            "The challenge is a scaled-up version of routine wizarding chess behaviour "
            "rather than a special artefact — which is the pattern for most of the "
            "obstacles guarding the Philosopher's Stone."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Why did Voldemort specifically hunt the Potters?",
        past_attempts="A prophecy told him to?",
        hint=(
            "The prophecy is part of it, but prophecies in this world can apply to "
            "more than one person. What did Voldemort actually choose when he decided "
            "which family to target?"
        ),
        next_question="What role does personal choice play in turning a prophecy into a destiny?",
        explanation=(
            "Prophecies gain their force through who acts on them. The Potters became "
            "the target because of Voldemort's interpretation — someone else could "
            "have fit the prophecy just as well."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="How is Fiendfyre different from normal fire?",
        past_attempts="none",
        hint=(
            "Think about what kind of magic Crabbe used it for, and why it destroyed "
            "a Horcrux when ordinary Incendio cannot. What quality makes Fiendfyre "
            "dangerous to its caster too?"
        ),
        next_question="What category of magic includes curses that don't simply obey the wand-wielder?",
        explanation=(
            "Fiendfyre sits in a category of magic that is less controlled than invoked. "
            "Its power to destroy unusual objects comes at the cost of being near-"
            "impossible to stop once released."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Why does Harry keep the Invisibility Cloak but return the Stone?",
        past_attempts="Is it because the Stone drove Cadmus mad?",
        hint=(
            "That's the right lesson from the Peverell story — but also think about "
            "what each Hallow offers to its owner and which offer is most dangerous to "
            "accept."
        ),
        next_question="Which of the three Hallows represents acceptance of death rather than denial?",
        explanation=(
            "The three objects map onto three attitudes toward mortality. Harry's "
            "choice reflects which attitude he adopts, not merely which object is "
            "most useful."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Why is Nagini harder to destroy than a typical Horcrux object?",
        past_attempts="none",
        hint=(
            "Most Horcruxes are inanimate — they sit and wait. What makes a living "
            "Horcrux different from a cup or a diadem?"
        ),
        next_question="Why would Voldemort keep his most mobile Horcrux physically close to him?",
        explanation=(
            "A living Horcrux introduces agency and mobility into what is normally a "
            "static target. This changes both the protection strategy and the "
            "destruction strategy."
        ),
    ).with_inputs("question", "past_attempts"),
]


EVALSET = [
    dspy.Example(
        question="Why did Dumbledore leave the Deluminator to Ron?",
        past_attempts="none",
        hint=(
            "Dumbledore gave gifts according to what each recipient would eventually "
            "need. What specific moment during the Horcrux hunt would a device that "
            "draws someone back be useful for?"
        ),
        next_question="What did Ron do during the hunt that would require exactly this kind of tool to undo?",
        explanation=(
            "The bequest looks ornamental but is diagnostic — Dumbledore anticipated "
            "the rupture and gave the remedy in advance."
        ),
    ).with_inputs("question", "past_attempts"),

    dspy.Example(
        question="Can a wizard survive having their soul split multiple times without harm?",
        past_attempts="I don't think so — doesn't it damage them?",
        hint=(
            "You're on the right track. The damage isn't only magical — what happens "
            "to personality, appearance, or reasoning when a soul is repeatedly torn?"
        ),
        next_question="What changes do we observe in Voldemort's appearance and behaviour over time that support this?",
        explanation=(
            "The cost of Horcrux creation is visible as well as spiritual. A wizard "
            "who has split their soul many times becomes progressively less human in "
            "both form and judgement."
        ),
    ).with_inputs("question", "past_attempts"),
]

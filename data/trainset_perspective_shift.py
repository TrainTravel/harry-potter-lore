"""
Training data — Perspective Shift mode
=======================================
Hand-labelled examples for the `perspective_shift` mode, where a modern
real-life scenario is re-framed through the lens of a specific HP character
whose canonical traits illuminate it.

Each example specifies:
  - scenario: a concrete, adult, real-life problem (1–3 sentences)
  - character: a character slug the mode will borrow as a lens
  - expected_citations: space-separated chunk_ids that good answers should
    ground the character's voice in (pattern: `{character}/{section}-NNN`)

The `answer` / reframe field is intentionally left for BootstrapFewShot to
generate — we only label what is cheap and objective (citations) so the
metric can check that the reframe is actually anchored in canon.

Valid character slugs (11):
    harry-potter, hermione-granger, ron-weasley, albus-dumbledore,
    severus-snape, minerva-mcgonagall, luna-lovegood, neville-longbottom,
    lord-voldemort, draco-malfoy, rubeus-hagrid

Citation chunk_ids come from data/character_lore_tagged.jsonl. Each is of the
form `{character-slug}/{section-slug}-NNN`; 1–3 per example is enough.
"""

from __future__ import annotations
import dspy


# ---------------------------------------------------------------------------
# Training set (used by BootstrapFewShot.compile)
# ---------------------------------------------------------------------------

TRAINSET = [
    # --- Career / identity paralysis ---
    dspy.Example(
        scenario="I'm 32, working a stable tech job that pays well but bores me, and I keep fantasising about quitting to write full-time. I can't tell if that's a real calling or just a fantasy to escape burnout.",
        character="albus-dumbledore",
        expected_citations="albus-dumbledore/personality-and-traits-001 albus-dumbledore/biography-aftermath-turning-down-power-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="I got a promotion I spent three years chasing and now I feel empty. Everyone congratulates me and I just want to disappear into the bathroom.",
        character="harry-potter",
        expected_citations="harry-potter/personality-and-traits-001 harry-potter/biography-early-life-discovery-of-being-a-wizard-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="I want to leave my finance career to become a wildlife vet. I'd be starting from zero at 38 and my partner thinks I'm having a crisis.",
        character="rubeus-hagrid",
        expected_citations="rubeus-hagrid/personality-and-traits-001",
    ).with_inputs("scenario", "character"),

    # --- Relationships: romantic / unrequited ---
    dspy.Example(
        scenario="I've been in love with my best friend for seven years. She's getting married next month and I'm the best man. I don't know how to hold this without it poisoning me.",
        character="severus-snape",
        expected_citations="severus-snape/relationships-lily-evans-001 severus-snape/personality-and-traits-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="My partner and I fight the same fight every three months: they say I work too much, I say they don't understand how important my career is. Neither of us is wrong and it's destroying us.",
        character="ron-weasley",
        expected_citations="ron-weasley/relationships-family-hermione-granger-001 ron-weasley/personality-and-traits-001",
    ).with_inputs("scenario", "character"),

    # --- Relationships: friendships ---
    dspy.Example(
        scenario="My closest friend is leaving our shared city for a better job. I'm happy for them and also furious, and the guilt of that fury is eating me.",
        character="ron-weasley",
        expected_citations="ron-weasley/relationships-family-harry-potter-001 ron-weasley/personality-and-traits-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="My friend group adopted a new person who is everything I am, but louder and funnier. I've started avoiding the group chat.",
        character="ron-weasley",
        expected_citations="ron-weasley/personality-and-traits-001 ron-weasley/relationships-family-harry-potter-001",
    ).with_inputs("scenario", "character"),

    # --- Grief / loss ---
    dspy.Example(
        scenario="My dad died three months ago. Everyone keeps asking 'how I'm doing' and I've started giving a pre-recorded answer because the real one takes too long.",
        character="harry-potter",
        expected_citations="harry-potter/biography-early-life-attack-at-godric-s-hollow-1981-001 harry-potter/personality-and-traits-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="I lost my sister last year in an accident I sometimes think I could have prevented. I haven't told anyone the second part.",
        character="albus-dumbledore",
        expected_citations="albus-dumbledore/relationships-family-ariana-dumbledore-001 albus-dumbledore/biography-romance-and-tragedy-002",
    ).with_inputs("scenario", "character"),

    # --- Impostor syndrome / overwork ---
    dspy.Example(
        scenario="I got into a top-tier graduate program and I've been awake for 48 hours convinced they'll rescind the offer when they realise the mistake.",
        character="hermione-granger",
        expected_citations="hermione-granger/personality-and-traits-001 hermione-granger/biography-early-life-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="I keep working 70-hour weeks to prove I 'belong' on my team even though no one has ever questioned my competence except me.",
        character="hermione-granger",
        expected_citations="hermione-granger/personality-and-traits-001 hermione-granger/biography-hogwarts-years-fourth-year-society-for-the-promotion-of-elfish-welfare-001",
    ).with_inputs("scenario", "character"),

    # --- Being different / belonging ---
    dspy.Example(
        scenario="I'm neurodivergent and just left a workplace that kept 'gently' telling me my energy was 'a lot'. I'm scared every new job will do the same.",
        character="luna-lovegood",
        expected_citations="luna-lovegood/personality-and-traits-001 luna-lovegood/personality-and-traits-luna-s-beliefs-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="I moved back to my hometown after a decade in a big city and I can feel my old friends deciding I've become 'pretentious'. I haven't — I've just changed.",
        character="luna-lovegood",
        expected_citations="luna-lovegood/personality-and-traits-001 luna-lovegood/biography-hogwarts-years-early-years-001",
    ).with_inputs("scenario", "character"),

    # --- Toxic authority / power dynamics ---
    dspy.Example(
        scenario="My manager publicly mocks my ideas in meetings and then pitches them upward as his own two weeks later. HR knows. Nothing happens.",
        character="minerva-mcgonagall",
        expected_citations="minerva-mcgonagall/relationships-dolores-umbridge-001 minerva-mcgonagall/biography-second-wizarding-war-high-inquisitor-at-hogwarts-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="A senior colleague keeps 'mentoring' me in ways that feel more like emotional surveillance. I can't name why it's wrong, but I dread every one-on-one.",
        character="severus-snape",
        expected_citations="severus-snape/personality-and-traits-001 severus-snape/relationships-albus-dumbledore-001",
    ).with_inputs("scenario", "character"),

    # --- Standing up to family expectations ---
    dspy.Example(
        scenario="My parents want me to take over the family business. I'd rather go into nursing. Every dinner has become a rehearsal of the same unspoken argument.",
        character="neville-longbottom",
        expected_citations="neville-longbottom/personality-and-traits-001 neville-longbottom/biography-early-life-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="I came out to my mother last year. She 'still loves me' but has quietly stopped inviting my partner to anything. I don't know whether to confront her or just let the silence keep growing.",
        character="neville-longbottom",
        expected_citations="neville-longbottom/personality-and-traits-001 neville-longbottom/relationships-family-augusta-longbottom-001",
    ).with_inputs("scenario", "character"),

    # --- Ethics / moral dilemmas ---
    dspy.Example(
        scenario="I found evidence my company is misleading regulators about emissions. Reporting it probably ends my career. Not reporting it definitely ends my self-respect.",
        character="albus-dumbledore",
        expected_citations="albus-dumbledore/personality-and-traits-001 albus-dumbledore/biography-second-wizarding-war-training-harry-potter-001",
    ).with_inputs("scenario", "character"),

    dspy.Example(
        scenario="A friend confessed something to me in confidence that, if I keep it secret, will hurt someone else I care about. I don't see a clean choice.",
        character="severus-snape",
        expected_citations="severus-snape/personality-and-traits-001 severus-snape/relationships-albus-dumbledore-001",
    ).with_inputs("scenario", "character"),

    # --- Money / class anxiety ---
    dspy.Example(
        scenario="I'm the only one in my friend group who can't afford the annual group holiday. Every year I invent a different excuse. I'm running out of excuses.",
        character="ron-weasley",
        expected_citations="ron-weasley/biography-early-life-001 ron-weasley/personality-and-traits-001",
    ).with_inputs("scenario", "character"),

    # --- Fame / unwanted attention ---
    dspy.Example(
        scenario="A post I wrote went viral and I now have 40k followers I didn't ask for. Half send me marriage proposals and half send me death threats. I just wanted to share a thought.",
        character="harry-potter",
        expected_citations="harry-potter/personality-and-traits-001 harry-potter/biography-hogwarts-years-fourth-year-1994-1995-mad-eye-moody-001",
    ).with_inputs("scenario", "character"),

    # --- Trauma / healing ---
    dspy.Example(
        scenario="I left an emotionally abusive relationship a year ago. I'm safe now but I still flinch when anyone raises their voice, even strangers on the street.",
        character="neville-longbottom",
        expected_citations="neville-longbottom/biography-early-life-001 neville-longbottom/personality-and-traits-001",
    ).with_inputs("scenario", "character"),

    # --- Late starts / career pivots ---
    dspy.Example(
        scenario="I spent my twenties being 'the quiet one' at every job. I'm 35 now and finally finding my voice and I feel ten years behind everyone else.",
        character="neville-longbottom",
        expected_citations="neville-longbottom/personality-and-traits-001 neville-longbottom/biography-hogwarts-years-seventh-year-battle-of-hogwarts-001",
    ).with_inputs("scenario", "character"),

    # --- Inherited values / peer pressure ---
    dspy.Example(
        scenario="I grew up in a deeply political household and just realised I've been repeating opinions for thirty years that I never actually examined. I don't know who I am underneath them.",
        character="draco-malfoy",
        expected_citations="draco-malfoy/personality-and-traits-001 draco-malfoy/relationships-family-parents-001",
    ).with_inputs("scenario", "character"),

    # --- Fear of death / obsession ---
    dspy.Example(
        scenario="I'm 44, healthy, and lie awake most nights thinking about my own death. I've started making strange decisions — skipping family events, hoarding money — to 'buy back time'.",
        character="lord-voldemort",
        expected_citations="lord-voldemort/personality-and-traits-001 lord-voldemort/biography-hogwarts-years-learning-about-horcruxes-001",
    ).with_inputs("scenario", "character"),
]


# ---------------------------------------------------------------------------
# Held-out evaluation set — do NOT use in BootstrapFewShot.compile
# ---------------------------------------------------------------------------

EVALSET = [
    # Career / identity paralysis
    dspy.Example(
        scenario="I've been offered a senior role at a company I don't respect. The money would let me buy my mother a house. I'd wake up every day already ashamed.",
        character="albus-dumbledore",
        expected_citations="albus-dumbledore/biography-aftermath-turning-down-power-001 albus-dumbledore/personality-and-traits-001",
    ).with_inputs("scenario", "character"),

    # Relationships / unrequited
    dspy.Example(
        scenario="My ex married someone new last month. I thought I was over it. I cried in a supermarket seeing his favourite cereal.",
        character="severus-snape",
        expected_citations="severus-snape/relationships-lily-evans-001",
    ).with_inputs("scenario", "character"),

    # Grief
    dspy.Example(
        scenario="My mother has early-onset dementia and already doesn't recognise me some days. I'm mourning someone who's still alive and no one has language for that.",
        character="harry-potter",
        expected_citations="harry-potter/personality-and-traits-001 harry-potter/biography-hogwarts-years-fifth-year-1995-1996-christmas-on-the-closed-ward-001",
    ).with_inputs("scenario", "character"),

    # Impostor syndrome
    dspy.Example(
        scenario="I'm the first person in my family to finish university. At work I still feel like I'm performing 'educated' in a language everyone else was born speaking.",
        character="hermione-granger",
        expected_citations="hermione-granger/biography-early-life-001 hermione-granger/personality-and-traits-001",
    ).with_inputs("scenario", "character"),

    # Being different / belonging
    dspy.Example(
        scenario="Everyone in my friend group is getting married and having babies. I don't want either and I've started feeling like I'm failing a test nobody told me I was taking.",
        character="luna-lovegood",
        expected_citations="luna-lovegood/personality-and-traits-001 luna-lovegood/biography-later-life-001",
    ).with_inputs("scenario", "character"),

    # Toxic authority / boundary-setting
    dspy.Example(
        scenario="I'm the head of a small team and my own boss keeps asking me to enforce policies I disagree with. My team is starting to see me as the enemy.",
        character="minerva-mcgonagall",
        expected_citations="minerva-mcgonagall/biography-early-career-at-hogwarts-001 minerva-mcgonagall/relationships-dolores-umbridge-001",
    ).with_inputs("scenario", "character"),

    # Inherited values / peer pressure
    dspy.Example(
        scenario="All my college friends are in crypto and keep mocking me for 'playing it safe'. I want out of the group chat but they're my oldest friends.",
        character="draco-malfoy",
        expected_citations="draco-malfoy/personality-and-traits-001 draco-malfoy/relationships-vincent-crabbe-and-gregory-goyle-001",
    ).with_inputs("scenario", "character"),

    # Loyalty to misunderstood things
    dspy.Example(
        scenario="I adopted a reactive rescue dog everyone keeps telling me to rehome. He's difficult and I love him and I can't tell if I'm being loyal or stubborn.",
        character="rubeus-hagrid",
        expected_citations="rubeus-hagrid/personality-and-traits-001",
    ).with_inputs("scenario", "character"),
]

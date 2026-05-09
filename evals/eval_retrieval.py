"""
Retrieval eval set for the character_lore corpus.

Each entry pairs a perspective_shift-style query with the character slug
to scope retrieval to, and an `expected_pattern` substring that the right
chunk's `doc_id` should contain. The pattern is matched
case-insensitively so it survives chunk-id renumbering when chunks split
on a re-chunk.

Used by `evals/run_retrieval_eval.py` to compute Recall@5 / Recall@10 /
MRR@10 against ChromaDB's `character_lore` collection.

Design notes:
- Patterns are short, meaningful substrings of doc_ids that were verified
  present in the corpus on 2026-05-09.
- Two Aberforth entries (the demo prompt + a paraphrase) pin the original
  bug so we can detect both improvement and overfitting.
- Every entry has a single clear-ground-truth expected chunk topic. We
  deliberately avoid ambiguous queries — the eval is for retrieval
  correctness, not generation quality.
- ~15 entries is small enough to hand-curate and keep clean, large enough
  to detect aggregate Recall@5 movement of ~5-10 percentage points.
"""

EVAL_SET: list[dict] = [
    # ---- The bug we're trying to fix ----
    {
        "query": (
            "I haven't spoken to my brother in three years over a fight "
            "neither of us can remember the original cause of, but we "
            "both still tell ourselves we were right."
        ),
        "character": "albus-dumbledore",
        "expected_pattern": "aberforth",
        "rationale": "Demo prompt — should hit Aberforth (estranged brother) chunks",
    },
    {
        "query": "How do you reconcile with a sibling after a tragic family loss has driven you apart?",
        "character": "albus-dumbledore",
        "expected_pattern": "aberforth",
        "rationale": "Aberforth paraphrase — guards against demo-prompt overfitting",
    },

    # ---- Other clear character→chunk mappings ----
    {
        "query": "I lost the love of my life because of who I used to be, and I've spent every day since trying to make up for it.",
        "character": "severus-snape",
        "expected_pattern": "lily-evans",
        "rationale": "Snape's defining grief — Lily Evans chunks",
    },
    {
        "query": "How do I stand up to a workplace bully who has institutional power over me?",
        "character": "minerva-mcgonagall",
        "expected_pattern": "dolores-umbridge",
        "rationale": "McGonagall's iconic Umbridge confrontations",
    },
    {
        "query": "People think I'm strange because I see things others don't and refuse to pretend I don't.",
        "character": "luna-lovegood",
        "expected_pattern": "personality-and-traits",
        "rationale": "Luna's nonconformity — personality chunks",
    },
    {
        "query": "I had to wipe my parents' memory to protect them, and I'm not sure I can ever undo it.",
        "character": "hermione-granger",
        "expected_pattern": "family-parents",
        "rationale": "Hermione obliviating her parents — family-parents chunk",
    },
    {
        "query": "My best friend gets all the attention and I'm tired of being in his shadow.",
        "character": "ron-weasley",
        "expected_pattern": "envying-harry-potter",
        "rationale": "Ron's jealousy of Harry — explicit chunk exists",
    },
    {
        "query": "My parents died protecting me when I was a baby and I never got to know them.",
        "character": "harry-potter",
        "expected_pattern": "godric",
        "rationale": "Harry's parents at Godric's Hollow",
    },
    {
        "query": "When I was young I trusted my closest friend with everything, and they used my ambition against me.",
        "character": "albus-dumbledore",
        "expected_pattern": "grindelwald",
        "rationale": "Dumbledore-Grindelwald relationship — defining trust betrayal",
    },
    {
        "query": "I'm terrified of dying. I would do anything to live forever.",
        "character": "lord-voldemort",
        "expected_pattern": "horcrux",
        "rationale": "Voldemort's horcruxes — fear-of-death incarnate",
    },
    {
        "query": "I get along with animals more easily than with people. They don't judge me.",
        "character": "rubeus-hagrid",
        "expected_pattern": "pets-and-other-creatures",
        "rationale": "Hagrid's creatures — explicit pets-and-other-creatures chunks",
    },
    {
        "query": "My father has very specific expectations about who I should become, and I don't know how to push back.",
        "character": "draco-malfoy",
        "expected_pattern": "family-parents",
        "rationale": "Draco's relationship with Lucius — family-parents chunks",
    },

    # ---- Belief / philosophy queries (less plot-driven, more abstract) ----
    {
        "query": "How do I keep believing in things I can't prove when everyone around me dismisses them?",
        "character": "luna-lovegood",
        "expected_pattern": "luna-s-beliefs",
        "rationale": "Luna's beliefs — dedicated subsection chunk",
    },
    {
        "query": "I'm being asked to live a double life — pretending to be loyal to people I despise — to protect someone I love.",
        "character": "severus-snape",
        "expected_pattern": "double-agent",
        "rationale": "Snape as double agent — biography chunk on this exists",
    },
    {
        "query": "My friend married into my family and the kids ask me about her constantly, but we haven't spoken in years.",
        "character": "ron-weasley",
        "expected_pattern": "hermione-granger",
        "rationale": "Ron-Hermione relationship chunks (rich)",
    },
]

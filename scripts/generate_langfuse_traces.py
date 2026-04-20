"""
Fire 100 creative prompts at the local /ask endpoint to populate Langfuse
traces. Rate-limited to stay under Gemini 2.5 Flash-lite's free-tier ceiling
(15 req/min, 1500/day).

Usage:
    # Start the server in another terminal first:
    #   source .env
    #   .venv/bin/uvicorn api.main:app --port 8000
    #
    # Then run this:
    .venv/bin/python scripts/generate_langfuse_traces.py
    .venv/bin/python scripts/generate_langfuse_traces.py --delay 3
    .venv/bin/python scripts/generate_langfuse_traces.py --only perspective_shift
    .venv/bin/python scripts/generate_langfuse_traces.py --n 20

Traces appear in Langfuse within ~1-2s of each call. Background flusher
batches events — no noticeable latency added to /ask.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import requests


# ---------------------------------------------------------------------------
# 100 question stims, mixed across 6 modes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stim:
    mode: str
    question: str
    extra: dict = None


STIMS: list[Stim] = [
    # ===== deep_research (18) =====
    Stim("deep_research", "Who killed Dumbledore and why?"),
    Stim("deep_research", "What are the three Deathly Hallows?"),
    Stim("deep_research", "How did Harry become master of the Elder Wand without ever wielding it?"),
    Stim("deep_research", "How many Horcruxes did Voldemort create and which were they?"),
    Stim("deep_research", "Trace the full history of the Elder Wand across its known owners."),
    Stim("deep_research", "Who founded the Order of the Phoenix and when?"),
    Stim("deep_research", "What happened during the Battle of Hogwarts?"),
    Stim("deep_research", "What creatures guard Gringotts and how?"),
    Stim("deep_research", "How does the Sorting Hat decide house placement?"),
    Stim("deep_research", "What is the Marauder's Map and who made it?"),
    Stim("deep_research", "Who is Sirius Black to Harry Potter?"),
    Stim("deep_research", "How does Polyjuice Potion work?"),
    Stim("deep_research", "What is the history of the Chamber of Secrets?"),
    Stim("deep_research", "What form does Harry's Patronus take and why?"),
    Stim("deep_research", "Where is Hogwarts located?"),
    Stim("deep_research", "How did Voldemort die in the final battle?"),
    Stim("deep_research", "What did Hermione Granger do after the Second Wizarding War?"),
    Stim("deep_research", "Why are Harry's and Voldemort's wands called brother wands?"),  # distractor

    # ===== guided_learning (16) =====
    Stim("guided_learning", "Can you teach me why Horcruxes are dangerous?"),
    Stim("guided_learning", "Help me understand the prophecy about Harry and Voldemort."),
    Stim("guided_learning", "Why does Dumbledore trust Snape even after Snape was a Death Eater?"),
    Stim("guided_learning", "Explain the significance of the Triwizard Tournament."),
    Stim("guided_learning", "What makes Gryffindor students different from Slytherin students?"),
    Stim("guided_learning", "How do house-elves fit into the wizarding world?"),
    Stim("guided_learning", "What role does love play in defeating Voldemort?"),
    Stim("guided_learning", "Why did Harry survive the Killing Curse as a baby?"),
    Stim("guided_learning", "What lessons can we learn from Severus Snape's life?"),
    Stim("guided_learning", "How does the Wizarding World relate to the Muggle world?"),
    Stim("guided_learning", "Why is Dumbledore's Army considered important?"),
    Stim("guided_learning", "How do wand allegiances work when an owner is disarmed?"),
    Stim("guided_learning", "Why does the Weasley family matter narratively?"),
    Stim("guided_learning", "What is the Fidelius Charm and when is it used?"),
    Stim("guided_learning", "How are Horcruxes created and destroyed?"),
    Stim("guided_learning", "What is the Statute of Secrecy?"),

    # ===== open_analysis (16) =====
    Stim("open_analysis", "Is Dumbledore a hero or a manipulator?"),
    Stim("open_analysis", "Why did Snape become who he was?"),
    Stim("open_analysis", "What does the series say about moral ambiguity?"),
    Stim("open_analysis", "Is Harry a typical chosen-one, or does the series subvert that trope?"),
    Stim("open_analysis", "How does the series frame class and privilege through the Malfoys?"),
    Stim("open_analysis", "What is the emotional arc of Draco Malfoy across seven books?"),
    Stim("open_analysis", "Why is Hermione's knowledge emphasized narratively?"),
    Stim("open_analysis", "What does Ron's abandonment during the Horcrux hunt say about friendship?"),
    Stim("open_analysis", "How does grief function in the series, particularly after Cedric's death?"),
    Stim("open_analysis", "Is Voldemort a tragedy or simply a villain?"),
    Stim("open_analysis", "What does the Mirror of Erised reveal about the characters who look into it?"),
    Stim("open_analysis", "How does the series treat prophecy and free will?"),
    Stim("open_analysis", "What is the thematic role of Hogwarts as a setting?"),
    Stim("open_analysis", "How does Molly Weasley killing Bellatrix recontextualize Molly's character?"),
    Stim("open_analysis", "What does the epilogue suggest about breaking or continuing cycles?"),
    Stim("open_analysis", "Why is the motif of hidden identity so central to the series?"),

    # ===== perspective_shift (18) =====
    Stim("perspective_shift", "I'm stuck between a safe job and a risky creative path.", {"character": "Dumbledore"}),
    Stim("perspective_shift", "How do I handle unrequited love after many years?", {"character": "Snape"}),
    Stim("perspective_shift", "I'm anxious about always being the smartest in the room.", {"character": "Hermione"}),
    Stim("perspective_shift", "Someone on my team is slacking off and bringing everyone down.", {"character": "McGonagall"}),
    Stim("perspective_shift", "I feel like nobody understands me at my new workplace.", {"character": "Luna"}),
    Stim("perspective_shift", "My best friend betrayed my trust over something small.", {"character": "Hagrid"}),
    Stim("perspective_shift", "I'm exhausted from always being the one people rely on.", {"character": "Harry"}),
    Stim("perspective_shift", "I feel overshadowed by my more successful siblings.", {"character": "Ron"}),
    Stim("perspective_shift", "Should I forgive someone who deeply hurt me years ago?", {"character": "Dumbledore"}),
    Stim("perspective_shift", "How do I live with the weight of past regrets?", {"character": "Snape"}),
    Stim("perspective_shift", "How do I balance professional ambition with personal relationships?", {"character": "Hermione"}),
    Stim("perspective_shift", "An institution I work for is resistant to change I believe in.", {"character": "McGonagall"}),
    Stim("perspective_shift", "My coworkers mock me behind my back for my interests.", {"character": "Luna"}),
    Stim("perspective_shift", "I care too much about everyone and it's hurting me.", {"character": "Hagrid"}),
    Stim("perspective_shift", "I reached a major goal and now I feel completely empty.", {"character": "Harry"}),
    Stim("perspective_shift", "I'm afraid of disappointing my family with my career choices.", {"character": "Ron"}),
    Stim("perspective_shift", "My partner and I want different things for the future.", {"character": "Dumbledore"}),
    Stim("perspective_shift", "I'm dealing with imposter syndrome in a new leadership role.", {"character": "McGonagall"}),

    # ===== debate (16) =====
    Stim("debate", "Was Dumbledore right to hide Harry's fate from him until the end?"),
    Stim("debate", "Is Snape a hero or a villain?"),
    Stim("debate", "Should Harry have kept the Elder Wand instead of returning it?"),
    Stim("debate", "Was Draco Malfoy redeemable by the end of the series?"),
    Stim("debate", "Is Voldemort a coherent villain or a weak antagonist?"),
    Stim("debate", "Did the Order of the Phoenix make a meaningful difference in the war?"),
    Stim("debate", "Was Ron's abandonment during the Horcrux hunt justifiable?"),
    Stim("debate", "Should the Hogwarts house system be abolished?"),
    Stim("debate", "Was the Statute of Secrecy ultimately a mistake?"),
    Stim("debate", "Is Dumbledore actually more manipulative than Voldemort?"),
    Stim("debate", "Was Hagrid a responsible adult figure for Harry?"),
    Stim("debate", "Would the wizarding world be better if Muggles knew it existed?"),
    Stim("debate", "Is Hermione Granger smarter than Albus Dumbledore?"),
    Stim("debate", "Was the Ministry of Magic salvageable post-war?"),
    Stim("debate", "Was Snape's love for Lily his defining motivation above all else?"),
    Stim("debate", "Did Harry succeed because of luck or skill?"),

    # ===== satirical_podcast (16) =====
    Stim("satirical_podcast", "The Ministry of Magic's HR department", {"modern_angle": "corporate HR absurdity"}),
    Stim("satirical_podcast", "Quidditch commentary", {"modern_angle": "bored sports pundits"}),
    Stim("satirical_podcast", "Hogwarts Yelp reviews by rejected applicants", {"modern_angle": "Yelp reviews"}),
    Stim("satirical_podcast", "Dumbledore as a LinkedIn influencer", {"modern_angle": "LinkedIn thought leadership"}),
    Stim("satirical_podcast", "Hogwarts Letter rejection process", {"modern_angle": "college admissions"}),
    Stim("satirical_podcast", "Gringotts and wizarding banking", {"modern_angle": "crypto bros"}),
    Stim("satirical_podcast", "Diagon Alley gentrification", {"modern_angle": "neighborhood gentrification"}),
    Stim("satirical_podcast", "The Dark Mark as startup branding", {"modern_angle": "startup branding"}),
    Stim("satirical_podcast", "Potions class as grad school labor", {"modern_angle": "grad school adjunct life"}),
    Stim("satirical_podcast", "Horcruxes as a business model", {"modern_angle": "SaaS subscriptions"}),
    Stim("satirical_podcast", "The Quibbler as a newsletter", {"modern_angle": "Substack newsletter culture"}),
    Stim("satirical_podcast", "House-elves as gig workers", {"modern_angle": "gig economy"}),
    Stim("satirical_podcast", "Owl Post vs modern delivery services", {"modern_angle": "last-mile logistics"}),
    Stim("satirical_podcast", "Voldemort's reputation management", {"modern_angle": "PR crisis comms"}),
    Stim("satirical_podcast", "Hogwarts alumni networking", {"modern_angle": "alumni fundraising calls"}),
    Stim("satirical_podcast", "Dating in the wizarding world", {"modern_angle": "dating apps"}),
]

assert len(STIMS) == 100, f"expected 100 stims, got {len(STIMS)}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def fire_one(base_url: str, stim: Stim, timeout: int = 60) -> tuple[bool, str]:
    body = {"question": stim.question, "mode": stim.mode}
    if stim.extra:
        body.update(stim.extra)
    try:
        r = requests.post(f"{base_url}/ask", json=body, timeout=timeout)
        r.raise_for_status()
        return True, (r.json().get("answer") or "")[:80]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:120]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--delay", type=float, default=4.5,
                    help="Seconds between requests (default 4.5 stays under 15 req/min)")
    ap.add_argument("--only", default=None,
                    help="Filter to a single mode (e.g. 'perspective_shift')")
    ap.add_argument("--n", type=int, default=None,
                    help="Stop after N requests (default: all 100)")
    args = ap.parse_args()

    stims = STIMS
    if args.only:
        stims = [s for s in stims if s.mode == args.only]
        if not stims:
            print(f"No stims match mode={args.only!r}", file=sys.stderr)
            return 2
    if args.n:
        stims = stims[: args.n]

    print(f"Firing {len(stims)} prompts at {args.base_url}  delay={args.delay}s")
    t0 = time.time()
    ok = 0
    fail = 0
    for i, s in enumerate(stims, 1):
        tag = f"[{i:>3}/{len(stims)}] {s.mode:<18}"
        t_start = time.time()
        success, preview = fire_one(args.base_url, s)
        dt_ms = (time.time() - t_start) * 1000
        if success:
            ok += 1
            print(f"{tag}  {dt_ms:>6.0f}ms  ✓  {preview}")
        else:
            fail += 1
            print(f"{tag}  {dt_ms:>6.0f}ms  ✗  {preview}")
        if i < len(stims):
            time.sleep(args.delay)

    total = time.time() - t0
    print(f"\nDone in {total:.0f}s.  ok={ok}  fail={fail}")
    print(f"Check https://cloud.langfuse.com  →  Traces")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

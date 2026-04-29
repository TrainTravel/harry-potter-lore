# Lovable Follow-up Prompt: Three Gaps from First Regen

> Paste the prompt below into Lovable to fix the three pieces the previous
> regen missed: per-response mode badge, empty-chat suggestion chips, and
> stable test selectors. Tested DOM (2026-04-26) confirms none of these are
> currently rendered.

---

```
The previous regen kept the existing UI but missed three changes I asked for. Please add them now.

### Gap 1 — Render a per-response mode badge

The POST /ask response JSON includes two fields per turn:
- `routed_mode`: string like "deep_research", "perspective_shift", "debate", "none"
- `router_confidence`: string — "low", "medium", or "high"

Right now the UI ignores both fields. After each assistant message, render a small badge that reads `routed_mode` from THAT message's response (not from any global state, not from the conversation's selected mode).

Map `routed_mode` → label:
- deep_research → "Lore Lookup"
- guided_learning → "Tutoring"
- exam_grader → "Grading"
- open_analysis → "Analysis"
- perspective_shift → "Character"
- debate → "Debate"
- satirical_podcast → "Comedy"
- none → "Off-topic"

Color-code by `router_confidence`:
- high → green/success
- medium → amber/warning
- low → red/destructive

Keep the badge visually subtle — small chip below the message, not a header. Do NOT replace the existing conversation-row mode label; this is a per-message indicator.

### Gap 2 — Empty-chat suggestion chips

When the chat has no messages yet, render 4–6 clickable example prompts as chips. Clicking one populates the input field. Examples:

- "What are the Deathly Hallows?"
- "Sort me into a Hogwarts house"
- "Was Snape really a hero? Argue both sides"
- "Help me understand how Horcruxes work"
- "What would Dumbledore say about my career change?"
- "Do a podcast about Quidditch as an extreme sport"

These should disappear once the user sends their first message.

### Gap 3 — Stable test selectors

For automated end-to-end tests, attach `data-testid` attributes to:
- The mode badge from Gap 1: `data-testid="mode-badge"`
- Each assistant message container: `data-testid="assistant-message"`
- Each suggestion chip from Gap 2: `data-testid="suggestion-chip"`

These are invisible — they don't affect styling.

### Verification

After regenerating, the page should:
1. On empty chat, show 4–6 suggestion chip buttons.
2. After sending "What are the Deathly Hallows?", show a small "Lore Lookup" badge next to or below the assistant response, color-coded green (since the API will return `router_confidence: "high"`).
3. `document.querySelectorAll('[data-testid="mode-badge"]').length` should be ≥ 1 after a response.
```

---

## Why this prompt is shorter than the original

- It assumes the original regen landed (chat shape, response rendering, mode list).
- It lists only the three concrete DOM changes still missing.
- It includes a verification snippet so Lovable can self-check before claiming done.

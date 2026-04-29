# Lovable Prompt: Intent-Routing Chat UX

> Paste the prompt below into Lovable to redesign the chat interface around
> automatic intent routing (`mode=auto`).

---

```
Redesign the chat interface to use automatic intent routing. Here are the changes:

### 1. Remove the mode picker from the main chat flow

- Remove any mode selector/dropdown that users see before sending a message.
- Always send `"mode": "auto"` in the POST /ask request body.
- The user just types their question and hits send — no mode selection needed.

### 2. Show the routed mode as a badge AFTER the response

The API response now includes two new fields when mode="auto":
- `routed_mode`: string — the mode the router picked (e.g. "perspective_shift", "debate", "deep_research", "none")
- `router_confidence`: string — "low", "medium", or "high"

After each AI response, show a small subtle chip/badge below the message with:
- The routed_mode as a human-readable label (use this mapping):
  - deep_research → "Lore Lookup"
  - guided_learning → "Tutoring"
  - exam_grader → "Grading"
  - open_analysis → "Analysis"
  - perspective_shift → "Character"
  - debate → "Debate"
  - satirical_podcast → "Comedy"
  - none → "Off-topic"
- Color-code the confidence: high=green, medium=amber, low=red (subtle, not distracting)
- The badge should be small and secondary — don't make it visually prominent

### 3. Add a "capabilities" hint for new users

When the chat is empty (no messages yet), show a welcome area with 4-6 example prompts as clickable suggestion chips. Examples:
- "What are the Deathly Hallows?"
- "Sort me into a Hogwarts house"
- "Was Snape really a hero? Argue both sides"
- "Help me understand how Horcruxes work"
- "What would Dumbledore say about my career change?"
- "Do a podcast about Quidditch as an extreme sport"

Each chip should populate the input field when clicked. This teaches users what the agent can do without them needing to read a mode list.

### 4. Handle the "none" (off-topic) response gracefully

When `routed_mode` is "none", the API returns a capability list as the answer. Style this response differently — maybe with a lighter background or an info icon — so it feels like a helpful nudge rather than an error.

### 5. Optional: power-user mode override

Add a small icon button (e.g. a sliders/tune icon) next to the input field that opens a dropdown to force a specific mode. Default label: "Auto". This is tucked away for power users, not the primary flow.

### 6. Add stable test selectors

To support automated end-to-end testing, attach `data-testid` attributes to the
two elements the test suite asserts on:

- The mode badge (from §2): `data-testid="mode-badge"`
- Each assistant message container: `data-testid="assistant-message"`

These don't change the visual design — they're just stable hooks for Playwright.

### 7. API request shape reference

POST /ask
{
  "question": "Put me into a house!",
  "mode": "auto",
  "conversation_id": "<uuid for multi-turn>",
  "character": "Dumbledore",        // only used if mode is forced to perspective_shift
  "student_answer": "",             // only used for exam_grader
  "modern_angle": "modern life"     // only used for satirical_podcast
}

Response includes:
{
  "turn_id": "...",
  "answer": "...",
  "citations": [...],
  "cost_usd": 0.001,
  "latency_ms": 1200,
  "routed_mode": "perspective_shift",   // NEW — show as badge
  "router_confidence": "high"           // NEW — color-code the badge
}
```

---

## Verification checklist

After pasting the prompt into Lovable and it regenerates the UI:

1. Send "Put me into a house" — should work (routed to `perspective_shift` + Sorting Hat)
2. Send "What are the Deathly Hallows?" — should show "Lore Lookup" badge
3. Send "What's the weather?" — should show off-topic response styled as info nudge
4. Check that suggestion chips appear on empty chat
5. Check that the power-user mode override dropdown works

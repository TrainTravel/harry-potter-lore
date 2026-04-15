# Architecture Decision Records

This directory captures decisions about the architecture of the
`harry-potter-lore` / `context_harness` codebase. Each ADR documents a
single decision: the context that forced it, the chosen approach,
the consequences, and the alternatives we rejected.

## When to write an ADR

Write one when a change:

- Affects multiple modules or cross-cutting concerns
- Has a rejected alternative that future-you will ask about in six months
- Locks in a trade-off that is expensive to reverse

Skip one when:

- The implementation is self-evident from the code
- There is only one reasonable way to do it
- The change is reversible in an afternoon

## Index

| # | Title | Status |
|---|---|---|
| [0001](./0001-async-first-observability.md) | Async-first observability (fire-and-forget writes) | Accepted |

Future candidates (unwritten — backfill on demand, not speculatively):

- Reliability layering order (timeout inside circuit breaker inside retry)
- Consolidate on ChromaDB (drop FAISS) for O(log N) deletion
- Async post-turn summarization — never block the user turn
- Index version guard — fail-fast on embedding model change
- DSPy two-mode pattern — Signatures → ChainOfThought → per-mode compiled JSONs
- Trainset strategy — label cheap-to-verify fields, bootstrap the rest
- Metric design — citation overlap (research) vs Socratic gate (learning)
- Artifact schema evolution — SemVer + migrator registry + hash integrity

## Process

1. Draft the ADR in this directory with the next available number.
2. Set status to `Proposed`; open a PR. Discussion happens on the PR.
3. On merge, change status to `Accepted`. Never edit the body after acceptance —
   superseding decisions get a new ADR that references the old one.
4. The PR that implements the decision references the ADR in its title or body
   (e.g. `Implements ADR-0001`).

## Template

```markdown
# ADR-NNNN: <Short title in imperative mood>

Status: Proposed | Accepted | Superseded by ADR-XXXX
Date: YYYY-MM-DD

## Context

What problem are we solving? What forces are at play? What constraints does the
codebase or runtime environment impose?

## Decision

What did we choose? One or two sentences, concrete enough that a reader can
map the decision to actual code.

## Consequences

What becomes easier as a result of this decision? What becomes harder? What
new failure modes does it introduce?

## Alternatives considered

What else did we look at, and why did we reject it? Each alternative in one
paragraph is enough.
```

## Style

- Imperative mood in titles (`Add fire-and-forget observability`, not `We added...`)
- One screen of markdown per ADR — if it's longer, split it
- Link to code with `file.py:line` references so readers can jump to the implementation
- Don't justify obvious choices ("we use Python because we wrote the code in Python")
- Do record trade-offs you made consciously, even if they seem obvious now

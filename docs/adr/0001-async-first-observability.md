# ADR-0001: Async-first observability (fire-and-forget writes)

Status: Accepted
Date: 2026-04-15

## Context

The agent emits two categories of observability data on every turn:

- **Traces** — sequence-numbered events (`TURN_START`, `RETRIEVAL`, `LLM_CALL`,
  `ERROR`) written by `context_harness.tracer.Tracer`
- **Cost events** — token usage and dollar cost per LLM call, written by
  `context_harness.cost_tracker.CostTracker`

Both persist to SQLite via the same process. The user-facing latency budget
for a turn is small — typically a few seconds, dominated by LLM round-trips
and retrieval. Adding two synchronous SQLite writes per event on the critical
path would add tens to low-hundreds of milliseconds of unpredictable tail
latency, growing as the trace gets longer.

Worse, a SQLite contention stall (e.g. `SQLITE_BUSY` under concurrent writers)
would block the turn indefinitely. Observability should never be able to
take down the thing it observes.

## Decision

Both `TraceStore.save()` and `CostTracker.record()` are scheduled via
`asyncio.create_task(...)` and return immediately. The caller does not `await`
the persistence — a fire-and-forget write runs on the event loop after the
turn is handed back to the user.

Implementation in `context_harness/tracer.py:Tracer.event` and
`context_harness/cost_tracker.py:CostTracker.record`.

## Consequences

**Gains**
- The user-facing turn latency is bounded by LLM + retrieval only, not by
  how much is being logged.
- A stalled or crashed observability store cannot stall or crash a turn.
- Backpressure is handled by the event loop rather than the caller.

**Costs**
- On process crash, the last N queued writes may be lost. This is acceptable
  for traces and cost events (they are diagnostic, not transactional) but
  would be unacceptable for, say, billing records.
- Errors raised inside the background task do not propagate to the caller.
  We log them and move on. A write failure is not surfaced to the user.
- Ordering between events is preserved (single event loop, FIFO tasks), but
  durability guarantees are weaker than a synchronous commit.

## Alternatives considered

**Synchronous writes.** Simplest to reason about — commit-then-return.
Rejected because it puts SQLite on the user's critical path and ties the
turn's tail latency to disk contention.

**In-memory only.** Keep traces in a ring buffer, never persist. Rejected
because replaying a failed turn is one of the main reasons traces exist;
losing them on restart defeats the purpose.

**Background thread with a queue.** A dedicated writer thread consuming from
`queue.Queue`. Rejected because the rest of the codebase is async-first; adding
a thread introduces a second concurrency model with its own shutdown and error
semantics, for no gain over `asyncio.create_task()`.

**Batch commits.** Flush N events at a time. Rejected for now as premature
optimisation — SQLite commit overhead is not the bottleneck at current
traffic. Revisit if traces become hot.

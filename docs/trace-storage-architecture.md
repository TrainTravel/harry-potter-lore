# Trace Storage Architecture: DuckDB, Iceberg, and Multi-Agent Scaling

## Current state

The project uses DuckDB as a local embedded trace store (`context_harness/tracer.py`).
Each agent turn produces a sequence of `TraceEvent` rows with hot columns
(`turn_id`, `kind`, `latency_ms`, `ts`) and a JSON payload column for the rest.
DuckDB's columnar engine makes aggregations like `PERCENTILE_CONT(0.95)` over
millions of events run in milliseconds.

## The problem at scale

When you move from one agent to many agents (or multi-agent topologies), four
questions arise:

1. **Per-agent debugging**: "What did agent X do on turn Y?" (point query)
2. **Cross-agent analytics**: "Which capability is slowest across all agents?" (aggregation)
3. **Cost accounting**: "How much did team Z's agents spend this week?" (rollup)
4. **Correlation**: "Agent A called Agent B — show me the full distributed trace" (join across agent boundaries)

## The architecture spectrum

```
Single agent                          Multi-agent fleet
    │                                         │
    ▼                                         ▼
┌──────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│  DuckDB  │    │  DuckDB +    │    │  DuckDB +      │    │ ClickHouse / │
│  (local  │    │  Parquet     │    │  Iceberg +     │    │ Snowflake    │
│   file)  │    │  (S3 export) │    │  S3/GCS        │    │ (managed)    │
└──────────┘    └──────────────┘    └────────────────┘    └──────────────┘
  you are                             sweet spot for
  here                                most agent systems
```

## Where Apache Iceberg fits

Apache Iceberg is a **table format** (not a database) that sits on top of
object storage (S3/GCS). It gives you:

- **ACID transactions on files in S3** — multiple writers don't corrupt
  each other.
- **Schema evolution** — add columns to trace events without rewriting
  history.
- **Time travel** — query traces from last Tuesday without maintaining
  snapshots yourself.
- **Partition pruning** — `WHERE agent_id = X AND ts > Y` only reads
  the relevant files.

DuckDB reads Iceberg natively via `iceberg_scan()`. Each agent writes
Parquet files into an Iceberg-managed table on S3, and any analyst (or
any other agent) can query the full fleet's traces with a local DuckDB
— no central database server needed.

## Three topologies

### 1. Centralized (one coordinator, N worker agents)

```
Agent-1 ──┐
Agent-2 ──┼──▶  S3/Iceberg table  ◀── DuckDB (coordinator queries all)
Agent-3 ──┘
```

Each agent writes its own partition. The coordinator aggregates across
all of them. Simplest model.

### 2. Decentralized (peer agents, no coordinator)

```
Agent-1 ◀──▶ Agent-2 ◀──▶ Agent-3
   │              │              │
   ▼              ▼              ▼
  S3/Iceberg (shared namespace, partitioned by agent_id)
```

Each agent writes AND reads. Agent-2 can query Agent-1's traces to
understand why a delegated subtask failed. Iceberg's ACID prevents
write conflicts between concurrent agents.

### 3. Isolated (agents don't share state)

```
Agent-1 → local DuckDB file
Agent-2 → local DuckDB file
Agent-3 → local DuckDB file
          │
          ▼  (periodic export)
   S3/Iceberg (aggregate view for ops team)
```

Best for privacy/isolation. Each agent is fully self-contained locally.
The ops team queries the Iceberg table for fleet-wide analytics on a
separate cadence.

## Schema changes needed for multi-agent

The current `trace_events` table uses `turn_id` which is per-turn,
per-agent. For multi-agent tracing, two columns are needed:

| Column | Purpose |
|---|---|
| `agent_id` | Identifies which agent produced the event. Iceberg partitions by this. |
| `trace_id` | Spans agent boundaries. When Agent-A calls Agent-B, both share a `trace_id`. This is what OpenTelemetry's W3C Trace Context propagates. |

The existing `turn_id` stays — it scopes events within a single agent's
turn. The hierarchy becomes: `trace_id` → `agent_id` → `turn_id` → `seq`.

## Migration path

| Stage | What changes | When |
|---|---|---|
| **Now** | Local DuckDB file, one agent, `turn_id` + `seq` | Current project state |
| **Add a second agent** | Add `agent_id` and `trace_id` columns. Still local DuckDB. | When you build a delegating agent (e.g. research → fact-check) |
| **Deploy multiple instances** | Each instance writes Parquet to S3. Add Iceberg catalog (e.g. REST catalog or AWS Glue). DuckDB reads from anywhere via `iceberg_scan()`. | When agents run on different machines |
| **Fleet at scale** | Consider ClickHouse or Snowflake for managed infra, or keep DuckDB + Iceberg for zero-server operation. | When you have >10 agents or need real-time dashboards |

## Parquet is not a trace store

Parquet is a **file format**, not a database. Important to understand
where it fits and where it doesn't.

### What Parquet is good at

- **Columnar compression**: trace events compress 10-50x because `kind`
  has ~10 distinct values, `agent_id` is highly repetitive. A GB of JSON
  traces becomes 20-50MB of Parquet.
- **Predicate pushdown**: `SELECT ... WHERE kind = 'llm_call'` only reads
  the `kind` column, skips everything else.
- **Batch analytics**: "p95 latency by capability over 30 days" scans
  billions of rows fast.

### What Parquet is bad at

- **Point queries**: "give me turn abc-123 right now" requires scanning
  files until you find it. No index. This is your most common debugging
  query.
- **Append**: you can't add a row to an existing Parquet file. You write
  a whole new file. For real-time trace ingestion this means buffering
  then flushing a new file every N seconds.
- **Freshness**: there's an inherent 10-60 second delay between event
  and queryability (the buffer-flush cycle).

### The correct two-tier architecture

Parquet is the **cold tier**, not the hot path:

```
Agent event
    |
    v
DuckDB (local file)          <-- hot: point queries, last 24h, sub-ms
    |
    |  (periodic export, e.g. hourly)
    v
S3/Parquet + Iceberg catalog <-- cold: fleet-wide analytics, months of history
    |
    v
DuckDB (reads Iceberg)       <-- analyst queries cold data from their laptop
```

Write to DuckDB in real-time (it handles append-only single-writer fine).
Periodically `COPY ... TO 's3://...' (FORMAT PARQUET)` for the cold tier.
Iceberg manages the file catalog so queries don't scan everything.

### When Parquet-only fails

Some teams try to skip the hot store: agent -> buffer -> Parquet on S3.
Fine for batch analytics but terrible for debugging because:

- "Show me what just happened" has 30-60s latency
- Point lookups are slow (grep through files)
- No transactions — a crash mid-flush loses the buffer

Don't do this. Keep a real database (DuckDB) for the hot path. Use
Parquet as the archival format.

### Summary by layer

| Layer | Format | Latency | Use case |
|---|---|---|---|
| Hot (real-time) | DuckDB file | <1ms point query | Debugging, live `/trace/{id}` |
| Cold (archive) | Parquet + Iceberg on S3 | seconds | Fleet analytics, cost reports, historical queries |

## Why DuckDB + Iceberg over alternatives

| Alternative | Pros | Why not (yet) |
|---|---|---|
| **SQLite** | Simple, everywhere | Column scans are slow; no native JSON; no concurrent writers; no analytics queries |
| **Postgres** | Rock-solid, great SQL | Needs a running server; overkill for append-only traces; worse compression than columnar |
| **Jaeger / Tempo** | Purpose-built tracing UIs, flame graphs | 14-day default retention; traces aren't durable training signal; not queryable with SQL |
| **ClickHouse** | Best-in-class for write-heavy analytics at scale | Needs a server or managed service; overkill for <10 agents |
| **Snowflake** | Fully managed, scales infinitely | Cost; vendor lock-in; latency for interactive queries |
| **DuckDB + Iceberg** | Embedded, zero servers, reads S3 natively, ACID multi-writer via Iceberg, columnar analytics | Single-writer locally (but Iceberg solves multi-writer on S3) |
| **Parquet alone** | Great compression, universal format | Not a database — no indexes, no point queries, no append, 30-60s ingestion lag |

## Querying across agents (example)

Once the schema has `agent_id` and `trace_id`, cross-agent queries
become natural SQL:

```sql
-- Which agent is the bottleneck in traces that took > 5 seconds?
SELECT agent_id, kind,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95
FROM iceberg_scan('s3://traces/agent_events')
WHERE trace_id IN (
    SELECT trace_id FROM iceberg_scan('s3://traces/agent_events')
    GROUP BY trace_id HAVING MAX(ts) - MIN(ts) > 5.0
)
GROUP BY agent_id, kind
ORDER BY p95 DESC;
```

This query runs from a laptop with DuckDB — no cluster, no server.
That's the architectural payoff.

package deeptutor

import cats.effect.{IO, Ref}
import cats.syntax.all.*
import io.circe.{Encoder, Decoder}
import io.circe.generic.semiauto.*
import io.circe.syntax.*

// =============================================================================
// Tracer Algebra — Skill 6: Evaluation and Observability
// =============================================================================
// "Trace backward through the logs to find the root cause." — IBM video
//
// Every event in a turn is assigned a monotonically increasing sequence number.
// This makes it possible to replay exactly what the agent did and at what point
// it went wrong — without re-running the query.

enum TraceEventKind derives Encoder.AsObject, Decoder:
  case TurnStart, TurnEnd, LlmCall, ToolCall, ToolResult
  case Retrieval, ContextBuilt, Escalation, Error, Security


final case class TraceEvent(
    turnId: String,
    seq: Int,
    kind: TraceEventKind,
    payload: Map[String, String],
    ts: Long,
    latencyMs: Option[Double],
) derives Encoder.AsObject, Decoder


// ---------------------------------------------------------------------------
// Algebra
// ---------------------------------------------------------------------------

trait TracerAlgebra[F[_]]:
  /** Record a single event. Returns the event (with assigned seq number). */
  def record(
      kind: TraceEventKind,
      payload: Map[String, String] = Map.empty,
      latencyMs: Option[Double] = None,
  ): F[TraceEvent]

  /** All events recorded so far, in sequence order. */
  def events: F[List[TraceEvent]]

  /** Human-readable dump for debugging. */
  def dump: F[String]

  /** Measure latency of fa and record a TraceEvent with the result. */
  def timed[A](kind: TraceEventKind, payload: Map[String, String] = Map.empty)(fa: F[A]): F[A]


// ---------------------------------------------------------------------------
// In-memory interpreter
// ---------------------------------------------------------------------------

object TracerAlgebra:

  def forTurn(turnId: String): IO[TracerAlgebra[IO]] =
    Ref.of[IO, List[TraceEvent]](List.empty).map { ref =>
      new TracerAlgebra[IO]:
        private val startMs = System.currentTimeMillis()

        def record(
            kind: TraceEventKind,
            payload: Map[String, String],
            latencyMs: Option[Double],
        ): IO[TraceEvent] =
          ref.modify { events =>
            val evt = TraceEvent(
              turnId    = turnId,
              seq       = events.size + 1,
              kind      = kind,
              payload   = payload,
              ts        = System.currentTimeMillis(),
              latencyMs = latencyMs,
            )
            (events :+ evt, evt)
          }

        def events: IO[List[TraceEvent]] = ref.get

        def dump: IO[String] =
          events.map { evts =>
            val header = s"=== Trace: turn_id=$turnId ==="
            val lines  = evts.map { e =>
              val lat = e.latencyMs.map(l => f" [${l}%.1fms]").getOrElse("")
              val payload = e.payload.take(3).map { case (k, v) => s"$k=${v.take(40)}" }.mkString(", ")
              f"  [${e.seq}%03d] ${e.kind}%-18s$lat $payload"
            }
            val footer = s"  Total elapsed: ${System.currentTimeMillis() - startMs}ms"
            (header +: lines :+ footer).mkString("\n")
          }

        def timed[A](kind: TraceEventKind, payload: Map[String, String])(fa: IO[A]): IO[A] =
          for
            start  <- IO(System.nanoTime())
            result <- fa
            latMs   = (System.nanoTime() - start) / 1_000_000.0
            _      <- record(kind, payload, Some(latMs))
          yield result
    }


// ---------------------------------------------------------------------------
// EvalMetrics algebra
// ---------------------------------------------------------------------------

final case class TurnRecord(
    turnId: String,
    sessionId: String,
    capability: String,
    success: Boolean,
    latencyMs: Double,
    costUsd: Double = 0.0,
    error: String = "",
)

final case class EvalSummary(
    totalTurns: Int,
    successRate: Double,
    p50LatencyMs: Double,
    p95LatencyMs: Double,
    avgCostUsd: Double,
) derives Encoder.AsObject

trait EvalMetricsAlgebra[F[_]]:
  def record(r: TurnRecord): F[Unit]
  def summary: F[EvalSummary]

object EvalMetricsAlgebra:

  def inMemory: IO[EvalMetricsAlgebra[IO]] =
    Ref.of[IO, List[TurnRecord]](List.empty).map { ref =>
      new EvalMetricsAlgebra[IO]:
        def record(r: TurnRecord): IO[Unit] = ref.update(_ :+ r)

        def summary: IO[EvalSummary] = ref.get.map { records =>
          if records.isEmpty then EvalSummary(0, 0, 0, 0, 0)
          else
            val latencies = records.map(_.latencyMs).sorted
            EvalSummary(
              totalTurns    = records.size,
              successRate   = records.count(_.success).toDouble / records.size,
              p50LatencyMs  = percentile(latencies, 50),
              p95LatencyMs  = percentile(latencies, 95),
              avgCostUsd    = records.map(_.costUsd).sum / records.size,
            )
        }

      private def percentile(sorted: List[Double], p: Int): Double =
        if sorted.isEmpty then 0.0
        else sorted(math.min((sorted.size * p / 100), sorted.size - 1))
    }

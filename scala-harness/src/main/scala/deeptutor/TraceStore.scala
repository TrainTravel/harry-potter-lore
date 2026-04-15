package deeptutor

import cats.effect.{IO, Ref}
import cats.syntax.all.*
import io.circe.syntax.*

// =============================================================================
// TraceStore Algebra — process-level event store for trace replay
// =============================================================================
// The per-turn TracerAlgebra holds events only for its own turn and is
// discarded at turn end.  TraceStoreAlgebra accumulates every turn's events
// for the lifetime of the process — enabling GET /trace/{turn_id} replay.
//
// This is the Scala equivalent of Python's TraceStore (SQLite-backed).
// Here we keep events in a Ref-backed in-memory map so no I/O is needed.
// Swap for a Doobie/SQLite interpreter in production without touching routes.

trait TraceStoreAlgebra[F[_]]:
  /** Persist all events from a completed turn. */
  def save(events: List[TraceEvent]): F[Unit]

  /** Retrieve events for a given turn in sequence order. */
  def getTurn(turnId: String): F[List[TraceEvent]]

  /** Return a summary of all known turns (most-recent-first). */
  def listTurns(limit: Int = 20): F[List[TurnSummary]]


final case class TurnSummary(
    turnId: String,
    eventCount: Int,
    firstTs: Long,
) derives io.circe.Encoder.AsObject


object TraceStoreAlgebra:

  /** In-memory interpreter backed by a Ref[Map[turnId → events]]. */
  def inMemory: IO[TraceStoreAlgebra[IO]] =
    Ref.of[IO, Map[String, List[TraceEvent]]](Map.empty).map { ref =>
      new TraceStoreAlgebra[IO]:

        def save(events: List[TraceEvent]): IO[Unit] =
          events.headOption match
            case None         => IO.unit
            case Some(first)  =>
              ref.update { m =>
                val existing = m.getOrElse(first.turnId, List.empty)
                m.updated(first.turnId, (existing ++ events).sortBy(_.seq))
              }

        def getTurn(turnId: String): IO[List[TraceEvent]] =
          ref.get.map(_.getOrElse(turnId, List.empty))

        def listTurns(limit: Int): IO[List[TurnSummary]] =
          ref.get.map { m =>
            m.values.toList
              .flatMap(evts => evts.headOption.map(first =>
                TurnSummary(first.turnId, evts.size, first.ts)
              ))
              .sortBy(-_.firstTs)
              .take(limit)
          }
    }

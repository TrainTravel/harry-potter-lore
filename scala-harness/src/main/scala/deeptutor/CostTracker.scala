package deeptutor

import cats.effect.{IO, Ref}
import cats.syntax.all.*
import io.circe.{Encoder, Decoder}
import io.circe.generic.semiauto.*

// =============================================================================
// Cost Tracker — Scala Typelevel (mirrors Python cost_tracker.py)
// Fixes deepTutor Weakness #5
// =============================================================================

// ---------------------------------------------------------------------------
// Domain
// ---------------------------------------------------------------------------

final case class PricingRate(inputPer1M: Double, outputPer1M: Double)

object PricingRate:
  val table: Map[String, PricingRate] = Map(
    "gemini-1.5-pro"    -> PricingRate(3.50,  10.50),
    "gemini-1.5-flash"  -> PricingRate(0.35,   1.05),
    "gemini-2.0-flash"  -> PricingRate(0.10,   0.40),
    "claude-opus-4"     -> PricingRate(15.00,  75.00),
    "claude-sonnet-4-6" -> PricingRate(3.00,   15.00),
    "claude-haiku-4-5"  -> PricingRate(0.80,    4.00),
    "gpt-4o"            -> PricingRate(2.50,   10.00),
    "gpt-4o-mini"       -> PricingRate(0.15,    0.60),
  )
  val default: PricingRate = PricingRate(1.00, 3.00)

  def estimateCostUsd(model: String, tokensIn: Int, tokensOut: Int): Double =
    val rate = table.getOrElse(model, default)
    (tokensIn * rate.inputPer1M + tokensOut * rate.outputPer1M) / 1_000_000.0


final case class CostEvent(
    model: String,
    capability: String,
    tokensIn: Int,
    tokensOut: Int,
    latencyMs: Double,
    sessionId: String,
    turnId: String,
    costUsd: Double,
    ts: Long,
) derives Encoder.AsObject, Decoder

object CostEvent:
  def apply(
      model: String,
      capability: String,
      tokensIn: Int,
      tokensOut: Int,
      latencyMs: Double,
      sessionId: String = "unknown",
      turnId: String = "unknown",
  ): CostEvent =
    CostEvent(
      model, capability, tokensIn, tokensOut, latencyMs,
      sessionId, turnId,
      costUsd = PricingRate.estimateCostUsd(model, tokensIn, tokensOut),
      ts = System.currentTimeMillis(),
    )


final case class CapabilityStats(
    capability: String,
    callCount: Int,
    totalTokensIn: Int,
    totalTokensOut: Int,
    totalCostUsd: Double,
    avgLatencyMs: Double,
) derives Encoder.AsObject

final case class CostSummary(
    totalEvents: Int,
    totalCostUsd: Double,
    byCapability: Map[String, CapabilityStats],
) derives Encoder.AsObject


// ---------------------------------------------------------------------------
// Algebra
// ---------------------------------------------------------------------------

trait CostTrackerAlgebra[F[_]]:
  def record(event: CostEvent): F[Unit]
  def summary: F[CostSummary]
  def sessionCost(sessionId: String): F[Double]

  /** Wrap an F[A] computation, measure latency, record cost. */
  def timed[A](
      capability: String,
      model: String,
      sessionId: String,
      turnId: String,
  )(fa: F[(Int, Int, A)]): F[A]   // fa returns (tokensIn, tokensOut, result)


// ---------------------------------------------------------------------------
// In-memory interpreter (Cats Effect Ref)
// ---------------------------------------------------------------------------

object CostTrackerAlgebra:

  def inMemory: IO[CostTrackerAlgebra[IO]] =
    Ref.of[IO, List[CostEvent]](List.empty).map { ref =>
      new CostTrackerAlgebra[IO]:

        def record(event: CostEvent): IO[Unit] =
          ref.update(_ :+ event)

        def summary: IO[CostSummary] =
          ref.get.map { events =>
            val byCapability = events
              .groupBy(_.capability)
              .map { case (cap, evts) =>
                cap -> CapabilityStats(
                  capability    = cap,
                  callCount     = evts.size,
                  totalTokensIn = evts.map(_.tokensIn).sum,
                  totalTokensOut= evts.map(_.tokensOut).sum,
                  totalCostUsd  = evts.map(_.costUsd).sum,
                  avgLatencyMs  = evts.map(_.latencyMs).sum / evts.size,
                )
              }
            CostSummary(
              totalEvents  = events.size,
              totalCostUsd = events.map(_.costUsd).sum,
              byCapability = byCapability,
            )
          }

        def sessionCost(sessionId: String): IO[Double] =
          ref.get.map(_.filter(_.sessionId == sessionId).map(_.costUsd).sum)

        def timed[A](
            capability: String,
            model: String,
            sessionId: String,
            turnId: String,
        )(fa: IO[(Int, Int, A)]): IO[A] =
          for
            start              <- IO(System.nanoTime())
            triple             <- fa
            (tIn, tOut, result) = triple
            latencyMs           = (System.nanoTime() - start) / 1_000_000.0
            _                  <- record(CostEvent(model, capability, tIn, tOut, latencyMs, sessionId, turnId))
          yield result
    }

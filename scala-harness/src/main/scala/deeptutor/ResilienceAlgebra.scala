package deeptutor

import cats.effect.{IO, Ref}
import cats.syntax.all.*

import scala.concurrent.duration.*

// =============================================================================
// Resilience Algebra — Skill 4: Reliability Engineering
// =============================================================================
// Mirrors Python reliability.py using purely functional Cats Effect primitives.
// Cats Effect's IO gives us structured concurrency and composable error handling
// for free — no mutable state needed.

// ---------------------------------------------------------------------------
// RetryPolicy
// ---------------------------------------------------------------------------

final case class RetryPolicy(
    maxAttempts: Int = 4,
    baseDelayMs: Long = 1000,
    maxDelayMs: Long = 30000,
    exponentialBase: Double = 2.0,
):
  /**
   * Execute fa with exponential-backoff retries.
   * Only retries on Throwable — caller decides which errors are retryable
   * by wrapping in a guard (see `retrying` below).
   */
  def execute[A](fa: IO[A]): IO[A] =
    def loop(attempt: Int): IO[A] =
      fa.handleErrorWith { err =>
        if attempt >= maxAttempts then IO.raiseError(RetryExhausted(attempt, err))
        else
          val cap = math.min(maxDelayMs, baseDelayMs * math.pow(exponentialBase, attempt - 1).toLong)
          val jitter = (math.random() * cap).toLong
          IO.sleep(jitter.millis) *> loop(attempt + 1)
      }
    loop(1)

  /** Only retry when the predicate holds — all other errors propagate immediately. */
  def retrying[A](retryOn: Throwable => Boolean)(fa: IO[A]): IO[A] =
    def loop(attempt: Int): IO[A] =
      fa.handleErrorWith { err =>
        if !retryOn(err) then IO.raiseError(err)
        else if attempt >= maxAttempts then IO.raiseError(RetryExhausted(attempt, err))
        else
          val cap = math.min(maxDelayMs, baseDelayMs * math.pow(exponentialBase, attempt - 1).toLong)
          val jitter = (math.random() * cap).toLong
          IO.sleep(jitter.millis) *> loop(attempt + 1)
      }
    loop(1)

case class RetryExhausted(attempts: Int, cause: Throwable)
    extends RuntimeException(s"All $attempts retry attempts exhausted: ${cause.getMessage}", cause)


// ---------------------------------------------------------------------------
// Circuit Breaker state
// ---------------------------------------------------------------------------

enum CircuitState:
  case Closed
  case Open(openedAt: Long)
  case HalfOpen


// ---------------------------------------------------------------------------
// CircuitBreaker algebra
// ---------------------------------------------------------------------------

trait CircuitBreakerAlgebra[F[_]]:
  def call[A](fa: F[A]): F[A]
  def state: F[CircuitState]
  def stats: F[CircuitStats]
  def reset: F[Unit]

final case class CircuitStats(
    state: String,
    failureCount: Int,
    successCount: Int,
)


object CircuitBreakerAlgebra:

  def inMemory(
      failureThreshold: Int = 5,
      recoveryTimeoutMs: Long = 60000,
      successThreshold: Int = 2,
  ): IO[CircuitBreakerAlgebra[IO]] =

    final case class CBState(
        circuit: CircuitState,
        failures: Int,
        successes: Int,
    )

    Ref.of[IO, CBState](CBState(CircuitState.Closed, 0, 0)).map { ref =>
      new CircuitBreakerAlgebra[IO]:

        def state: IO[CircuitState] = ref.get.map(resolveState)

        def stats: IO[CircuitStats] = ref.get.map { s =>
          CircuitStats(resolveState(s).toString, s.failures, s.successes)
        }

        def reset: IO[Unit] = ref.set(CBState(CircuitState.Closed, 0, 0))

        def call[A](fa: IO[A]): IO[A] =
          ref.get.map(resolveState).flatMap {
            case CircuitState.Open(_) =>
              IO.raiseError(CircuitOpenError("Circuit is OPEN — failing fast"))

            case _ =>
              fa.attempt.flatMap {
                case Right(result) =>
                  onSuccess.as(result)
                case Left(err) =>
                  onFailure *> IO.raiseError(err)
              }
          }

        private def resolveState(s: CBState): CircuitState =
          s.circuit match
            case CircuitState.Open(openedAt) =>
              val elapsed = System.currentTimeMillis() - openedAt
              if elapsed >= recoveryTimeoutMs then CircuitState.HalfOpen
              else s.circuit
            case other => other

        private val onSuccess: IO[Unit] = ref.update { s =>
          s.circuit match
            case CircuitState.HalfOpen =>
              val newSuccesses = s.successes + 1
              if newSuccesses >= successThreshold then
                CBState(CircuitState.Closed, 0, 0)
              else
                s.copy(successes = newSuccesses, failures = 0)
            case _ => s.copy(failures = 0)
        }

        private val onFailure: IO[Unit] = ref.update { s =>
          val newFailures = s.failures + 1
          if newFailures >= failureThreshold then
            CBState(CircuitState.Open(System.currentTimeMillis()), newFailures, 0)
          else
            s.copy(failures = newFailures, successes = 0)
        }
    }


case class CircuitOpenError(msg: String)
    extends RuntimeException(msg) with DomainError:
  val message: String = msg


// ---------------------------------------------------------------------------
// ResilientCaller — retry + circuit breaker + timeout composed
// ---------------------------------------------------------------------------

class ResilientCaller(
    retry: RetryPolicy = RetryPolicy(),
    cb: CircuitBreakerAlgebra[IO],
    timeoutMs: Long = 30000,
):
  def call[A](fa: IO[A]): IO[A] =
    retry.execute(
      cb.call(
        IO.race(fa, IO.sleep(timeoutMs.millis) *> IO.raiseError(new java.util.concurrent.TimeoutException(s"Timed out after ${timeoutMs}ms")))
          .flatMap {
            case Left(result) => IO.pure(result)
            case Right(_)     => IO.raiseError(new java.util.concurrent.TimeoutException())
          }
      )
    )

  def circuitStats: IO[CircuitStats] = cb.stats

package deeptutor

import cats.effect.{IO, Ref}
import cats.syntax.all.*

// =============================================================================
// Context Algebra — tagless-final style
// =============================================================================
// Typelevel pattern: define an *algebra* (trait) that describes *what* operations
// are available, then provide concrete *interpreters* (implementations).
//
// This decouples:
//   - business logic (uses the algebra)   from
//   - infrastructure (implements the algebra with real I/O)
// allowing you to swap implementations or test with stubs trivially.

// ---------------------------------------------------------------------------
// Algebra definition
// ---------------------------------------------------------------------------

trait ContextWindowAlgebra[F[_]]:
  /** Current token usage. */
  def usedTokens: F[Int]

  /** Remaining tokens in budget. */
  def remainingTokens: F[Int]

  /** Add an entry to the window, evicting if necessary. */
  def add(entry: ContextEntry): F[Unit]

  /** Add a text entry (convenience). */
  def addText(role: ContextRole, content: String, priority: Double = 1.0): F[ContextEntry]

  /** Remove all retrieval entries (between turns). */
  def clearRetrieval: F[Unit]

  /** Return a snapshot of all entries. */
  def entries: F[List[ContextEntry]]

  /** Render as {role, content} message list. */
  def toMessages: F[List[Map[String, String]]]

  /** Reset the entire window. */
  def reset: F[Unit]


// ---------------------------------------------------------------------------
// Token counter algebra (pluggable — swap in tiktoken, etc.)
// ---------------------------------------------------------------------------

trait TokenCounterAlgebra[F[_]]:
  def count(text: String): F[Int]

object TokenCounterAlgebra:
  /** Whitespace approximation — no I/O required. */
  def whitespace[F[_]: cats.Applicative]: TokenCounterAlgebra[F] =
    new TokenCounterAlgebra[F]:
      def count(text: String): F[Int] =
        text.split("\\s+").length.pure[F]


// ---------------------------------------------------------------------------
// In-memory interpreter
// ---------------------------------------------------------------------------

object ContextWindowAlgebra:

  def inMemory(
      maxTokens: Int = 4096,
      reservedOutput: Int = 512,
      counter: TokenCounterAlgebra[IO] = TokenCounterAlgebra.whitespace[IO],
  ): IO[ContextWindowAlgebra[IO]] =
    val budget = maxTokens - reservedOutput
    Ref.of[IO, List[ContextEntry]](List.empty).map { ref =>
      new ContextWindowAlgebra[IO]:

        def usedTokens: IO[Int] =
          ref.get.map(_.map(_.tokens).sum)

        def remainingTokens: IO[Int] =
          usedTokens.map(budget - _)

        def add(entry: ContextEntry): IO[Unit] =
          for
            tokens <- counter.count(entry.content)
            sized   = entry.copy(tokens = tokens)
            _      <- evictUntilFit(sized.tokens)
            _      <- ref.update(_ :+ sized)
          yield ()

        def addText(role: ContextRole, content: String, priority: Double): IO[ContextEntry] =
          for
            tokens <- counter.count(content)
            entry   = ContextEntry(role, content, tokens, priority)
            _      <- add(entry)
          yield entry

        def clearRetrieval: IO[Unit] =
          ref.update(_.filterNot(_.role == ContextRole.Retrieval))

        def entries: IO[List[ContextEntry]] = ref.get

        def toMessages: IO[List[Map[String, String]]] =
          ref.get.map(_.map(e => Map("role" -> e.role.toString.toLowerCase, "content" -> e.content)))

        def reset: IO[Unit] = ref.set(List.empty)

        // -- Eviction logic --------------------------------------------------
        private def evictUntilFit(needed: Int): IO[Unit] =
          for
            remaining <- remainingTokens
            _         <- if remaining >= needed then IO.unit
                         else evictOne.flatMap {
                           case true  => evictUntilFit(needed)
                           case false => IO.raiseError(ContextBudgetExceeded(budget - remaining + needed, budget))
                         }
          yield ()

        private def evictOne: IO[Boolean] =
          ref.get.map { es =>
            es.filterNot(_.role == ContextRole.System)
              .sortBy(e => (e.priority, e.metadata.getOrElse("ts", "0")))
              .headOption
          }.flatMap {
            case None         => IO.pure(false)
            case Some(victim) => ref.update(_.filterNot(_ eq victim)).as(true)
          }
    }

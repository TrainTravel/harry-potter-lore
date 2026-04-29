package deeptutor

import cats.effect.Sync
import cats.syntax.all.*

// =============================================================================
// AgentAlgebra + mode-specific agents (OpenAnalysis, PerspectiveShift).
// =============================================================================
// An agent runs the full loop for a single turn:
//   1. optional: load chat history for the conversation
//   2. retrieve grounding chunks from the vector store
//   3. render the prompt from the mode's SignatureSpec
//   4. call the LLM
//   5. parse the output
//   6. optional: persist the turn
// Errors are returned as Left(AgentError), never thrown.

// ---------------------------------------------------------------------------
// Typed errors — no exceptions for expected failures.
// ---------------------------------------------------------------------------

sealed trait AgentError:
  def message: String

object AgentError:
  final case class UnknownMode(mode: String) extends AgentError:
    val message = s"Unknown mode: $mode"

  final case class LLMFailed(underlying: LLMError) extends AgentError:
    val message = s"LLM failed: ${underlying.cause}"

  final case class ParseFailed(raw: String, missing: List[String]) extends AgentError:
    val message = s"Failed to parse output; missing fields: ${missing.mkString(", ")}"


// ---------------------------------------------------------------------------
// What an agent hands back.
// ---------------------------------------------------------------------------

final case class AgentOutput(
    turnIndex:   Option[Int],        // None for one-shot; Some(idx) for multi-turn
    fields:      Map[String, String],
    citations:   List[String],
    chunks:      List[Chunk],
)


// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

private[deeptutor] object AgentUtil:

  /** Build the "[doc_id] text" context block, or a stub string. */
  def formatContext(chunks: List[Chunk]): String =
    if chunks.isEmpty then "No context retrieved."
    else chunks.map(c => s"[${c.docId}] ${c.text}").mkString("\n\n")

  /** Parse a citations field into a list of bracketed tokens: "[a] [b]" -> List("[a]", "[b]"). */
  def parseCitations(raw: String): List[String] =
    """\[[^\]]+\]""".r.findAllIn(raw).toList

  /** Normalize a free-form character name to the canonical slug. */
  val characterSlug: Map[String, String] = Map(
    "harry"               -> "harry-potter",
    "harry potter"        -> "harry-potter",
    "hermione"            -> "hermione-granger",
    "hermione granger"    -> "hermione-granger",
    "ron"                 -> "ron-weasley",
    "ron weasley"         -> "ron-weasley",
    "dumbledore"          -> "albus-dumbledore",
    "albus"               -> "albus-dumbledore",
    "albus dumbledore"    -> "albus-dumbledore",
    "snape"               -> "severus-snape",
    "severus"             -> "severus-snape",
    "severus snape"       -> "severus-snape",
    "mcgonagall"          -> "minerva-mcgonagall",
    "minerva"             -> "minerva-mcgonagall",
    "minerva mcgonagall"  -> "minerva-mcgonagall",
    "luna"                -> "luna-lovegood",
    "luna lovegood"       -> "luna-lovegood",
    "neville"             -> "neville-longbottom",
    "neville longbottom"  -> "neville-longbottom",
    "voldemort"           -> "lord-voldemort",
    "lord voldemort"      -> "lord-voldemort",
    "tom riddle"          -> "lord-voldemort",
    "draco"               -> "draco-malfoy",
    "draco malfoy"        -> "draco-malfoy",
    "hagrid"              -> "rubeus-hagrid",
    "rubeus hagrid"       -> "rubeus-hagrid",
  )

  def normalizeCharacter(name: String): String =
    val key = Option(name).map(_.trim.toLowerCase).getOrElse("")
    characterSlug.getOrElse(key, key.replace(" ", "-"))


// ---------------------------------------------------------------------------
// OpenAnalysisAgent
// ---------------------------------------------------------------------------

final class OpenAnalysisAgent[F[_]: Sync] private (
    store: VectorStoreAlgebra[F],
    llm:   LLMClientAlgebra[F],
    conv:  ConversationStoreAlgebra[F],
):
  private val signature = SignatureSpec(
    instructions =
      """Analytical mode: use retrieved lore as a factual foundation, then draw on
        |your broader knowledge (psychology, literary theory, history, philosophy)
        |to provide deep analysis. Clearly mark which parts come from the corpus
        |vs your own reasoning. Do not refuse — if the corpus is thin, lean on
        |your general knowledge and say so.""".stripMargin,
    inputs = List(
      FieldSpec("question",     "the analytical question"),
      FieldSpec("context",      "retrieved lore passages, each prefixed [doc_id]"),
      FieldSpec("chat_history", "prior analysis turns in this conversation (may be empty)"),
    ),
    outputs = List(
      FieldSpec("analysis",      "3-5 sentences. Direct analysis blending corpus facts with broader knowledge."),
      FieldSpec("corpus_facts",  "2-3 sentences. ONLY facts drawn from [doc_id] chunks."),
      FieldSpec("own_reasoning", "2-3 sentences. Claims NOT in the corpus, marked as interpretation."),
      FieldSpec("citations",     "Space-separated doc_ids in [brackets], or 'none'."),
    ),
  )

  def ask(
      question:       String,
      conversationId: Option[ConversationId],
  ): F[Either[AgentError, AgentOutput]] =
    for
      history <- conversationId.fold(Sync[F].pure(""))(cid =>
                   conv.load(cid, 5).map(conv.formatForLlm(_, Mode.OpenAnalysis)))
      chunks  <- store.query(question, 7)
      inputs   = Map(
                   "question"     -> question,
                   "context"      -> AgentUtil.formatContext(chunks),
                   "chat_history" -> history,
                 )
      prompt   = signature.renderPrompt(inputs)
      result  <- llm.complete(prompt, systemOpt = None)
      out     <- result match
                   case Left(e) =>
                     Sync[F].pure(Left(AgentError.LLMFailed(e)): Either[AgentError, AgentOutput])
                   case Right(raw) =>
                     val fields = signature.parseOutput(raw)
                     val citations = AgentUtil.parseCitations(fields.getOrElse("citations", ""))
                     val turnOpt = conversationId.traverse { cid =>
                       conv.save(cid, Turn(
                         userMessage   = question,
                         agentResponse = fields,
                         mode          = Mode.OpenAnalysis,
                         character     = None,
                       ))
                     }
                     turnOpt.map { idx =>
                       Right(AgentOutput(idx, fields, citations, chunks)): Either[AgentError, AgentOutput]
                     }
    yield out


object OpenAnalysisAgent:
  def make[F[_]: Sync](
      store: VectorStoreAlgebra[F],
      llm:   LLMClientAlgebra[F],
      conv:  ConversationStoreAlgebra[F],
  ): OpenAnalysisAgent[F] = new OpenAnalysisAgent[F](store, llm, conv)


// ---------------------------------------------------------------------------
// PerspectiveShiftAgent
// ---------------------------------------------------------------------------

final class PerspectiveShiftAgent[F[_]: Sync] private (
    store: VectorStoreAlgebra[F],
    llm:   LLMClientAlgebra[F],
    conv:  ConversationStoreAlgebra[F],
):
  private val signature = SignatureSpec(
    instructions =
      """Extract a principle this Harry Potter character embodies, then apply it
        |to the user's real-world scenario. Ground everything in corpus facts,
        |then reason how those traits translate to practical advice.  Be
        |specific and actionable. Multi-turn: if `chat_history` is non-empty,
        |treat the current scenario as a continuation.""".stripMargin,
    inputs = List(
      FieldSpec("scenario",     "the real-world situation or question"),
      FieldSpec("character",    "the HP character whose perspective to apply"),
      FieldSpec("context",      "retrieved lore about this character, each prefixed [doc_id]"),
      FieldSpec("chat_history", "prior perspective-shift turns in this conversation"),
    ),
    outputs = List(
      // ORDER MATTERS — this is the reason-first / voice-last pattern.
      FieldSpec("character_principle",
        "2-3 sentences. Core principle the character embodies, anchored in a specific canon event."),
      FieldSpec("applied_insight",
        "3-4 sentences. Direct, actionable insight for the scenario."),
      FieldSpec("reasoning",
        "1-2 sentences. Why THIS character's experience maps to THIS scenario."),
      FieldSpec("character_response",
        "120-180 words. The character speaking DIRECTLY to the user, first person."),
      FieldSpec("citations",
        "Space-separated doc_ids in [brackets], using ONLY ids from context."),
    ),
  )

  def ask(
      scenario:       String,
      character:      String,
      conversationId: Option[ConversationId],
  ): F[Either[AgentError, AgentOutput]] =
    for
      history <- conversationId.fold(Sync[F].pure(""))(cid =>
                   conv.load(cid, 5).map(conv.formatForLlm(_, Mode.PerspectiveShift)))
      slug     = AgentUtil.normalizeCharacter(character)
      // Character-scoped retrieval first...
      scoped  <- store.retrieveWhere(scenario, 5, Map("character" -> slug))
      // ...fall back to generic when empty
      chunks  <- if scoped.nonEmpty then Sync[F].pure(scoped)
                 else store.query(s"$character $scenario", 5)
      inputs   = Map(
                   "scenario"     -> scenario,
                   "character"    -> character,
                   "context"      -> AgentUtil.formatContext(chunks),
                   "chat_history" -> history,
                 )
      prompt   = signature.renderPrompt(inputs)
      result  <- llm.complete(prompt, systemOpt = None)
      out     <- result match
                   case Left(e) =>
                     Sync[F].pure(Left(AgentError.LLMFailed(e)): Either[AgentError, AgentOutput])
                   case Right(raw) =>
                     val fields = signature.parseOutput(raw)
                     val citations = AgentUtil.parseCitations(fields.getOrElse("citations", ""))
                     val turnOpt = conversationId.traverse { cid =>
                       conv.save(cid, Turn(
                         userMessage   = scenario,
                         agentResponse = fields,
                         mode          = Mode.PerspectiveShift,
                         character     = Some(character),
                       ))
                     }
                     turnOpt.map { idx =>
                       Right(AgentOutput(idx, fields, citations, chunks)): Either[AgentError, AgentOutput]
                     }
    yield out


object PerspectiveShiftAgent:
  def make[F[_]: Sync](
      store: VectorStoreAlgebra[F],
      llm:   LLMClientAlgebra[F],
      conv:  ConversationStoreAlgebra[F],
  ): PerspectiveShiftAgent[F] = new PerspectiveShiftAgent[F](store, llm, conv)


// ---------------------------------------------------------------------------
// Top-level router
// ---------------------------------------------------------------------------

trait AgentAlgebra[F[_]]:
  /** Dispatch to a mode-specific agent. Errors returned as Left. */
  def ask(
      mode:           String,
      primary:        String,
      kwargs:         Map[String, String],
      conversationId: Option[ConversationId],
  ): F[Either[AgentError, AgentOutput]]


object AgentAlgebra:

  /** Wire the in-scope VectorStore + LLM + ConversationStore into both
    * ported agents and a dispatcher. */
  def wire[F[_]: Sync](
      store: VectorStoreAlgebra[F],
      llm:   LLMClientAlgebra[F],
      conv:  ConversationStoreAlgebra[F],
  ): AgentAlgebra[F] =
    val oa = OpenAnalysisAgent.make[F](store, llm, conv)
    val ps = PerspectiveShiftAgent.make[F](store, llm, conv)

    new AgentAlgebra[F]:
      def ask(
          mode:           String,
          primary:        String,
          kwargs:         Map[String, String],
          conversationId: Option[ConversationId],
      ): F[Either[AgentError, AgentOutput]] =
        mode match
          case "open_analysis" =>
            oa.ask(primary, conversationId)
          case "perspective_shift" =>
            val character = kwargs.getOrElse("character", "Dumbledore")
            ps.ask(primary, character, conversationId)
          case other =>
            Sync[F].pure(Left(AgentError.UnknownMode(other)))

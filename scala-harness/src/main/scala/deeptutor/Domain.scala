package deeptutor

import io.circe.{Decoder, Encoder}
import io.circe.generic.semiauto.*

// =============================================================================
// Domain model — pure data, no effects
// =============================================================================
// Context Engineering insight: making domain types explicit and total prevents
// a whole class of context injection bugs (wrong role, missing fields, etc.).

// ---------------------------------------------------------------------------
// Context roles — mirrors the Python ContextRole enum
// ---------------------------------------------------------------------------

enum ContextRole derives Encoder.AsObject, Decoder:
  case System
  case User
  case Assistant
  case Retrieval
  case Tool

// ---------------------------------------------------------------------------
// A single context entry
// ---------------------------------------------------------------------------

final case class ContextEntry(
    role: ContextRole,
    content: String,
    tokens: Int,
    priority: Double,
    metadata: Map[String, String] = Map.empty,
) derives Encoder.AsObject, Decoder

// ---------------------------------------------------------------------------
// A retrieved document chunk
// ---------------------------------------------------------------------------

final case class Chunk(
    chunkId: String,
    docId: String,
    text: String,
    score: Double,
    metadata: Map[String, String] = Map.empty,
) derives Encoder.AsObject, Decoder

// ---------------------------------------------------------------------------
// HTTP request / response shapes
// ---------------------------------------------------------------------------

final case class QueryRequest(
    query: String,
    topK: Int = 5,
    sessionId: Option[String] = None,
) derives Encoder.AsObject, Decoder

final case class QueryResponse(
    answer: String,
    chunks: List[Chunk],
    usedTokens: Int,
    sessionId: String,
) derives Encoder.AsObject, Decoder

final case class IngestRequest(
    docId: String,
    text: String,
    metadata: Map[String, String] = Map.empty,
) derives Encoder.AsObject, Decoder

final case class IngestResponse(
    docId: String,
    chunkCount: Int,
) derives Encoder.AsObject, Decoder

// ---------------------------------------------------------------------------
// Error model — typed errors keep the effect layer honest
// ---------------------------------------------------------------------------

sealed trait DomainError extends Throwable:
  def message: String
  override def getMessage: String = message

case class ContextBudgetExceeded(used: Int, budget: Int) extends DomainError:
  val message = s"Context budget exceeded: $used / $budget tokens used"

case class ChunkNotFound(chunkId: String) extends DomainError:
  val message = s"Chunk not found: $chunkId"

case class EmbeddingError(cause: String) extends DomainError:
  val message = s"Embedding failed: $cause"

case class LLMError(cause: String) extends DomainError:
  val message = s"LLM call failed: $cause"

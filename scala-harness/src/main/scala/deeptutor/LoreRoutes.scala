package deeptutor

import cats.effect.IO
import cats.syntax.all.*
import io.circe.syntax.*
import org.http4s.*
import org.http4s.circe.*
import org.http4s.circe.CirceEntityDecoder.*
import org.http4s.dsl.io.*
import org.typelevel.log4cats.Logger
import java.util.UUID
import fs2.Stream

// =============================================================================
// Http4s Routes — the HTTP face of the harness
// =============================================================================
// Context Engineering insight: the HTTP layer is *thin*.  It decodes requests,
// delegates to the pipeline, and encodes responses.  No business logic lives here.

class LoreRoutes(store: VectorStoreAlgebra[IO])(using logger: Logger[IO]):

  // Per-request window factory (in production: session-keyed map)
  private def freshWindow: IO[ContextWindowAlgebra[IO]] =
    ContextWindowAlgebra.inMemory(maxTokens = 4096, reservedOutput = 512)

  val routes: HttpRoutes[IO] = HttpRoutes.of[IO] {

    // -------------------------------------------------------------------------
    // POST /ingest  — add lore text to the vector store
    // -------------------------------------------------------------------------
    case req @ POST -> Root / "ingest" =>
      for
        body    <- req.as[IngestRequest]
        _       <- logger.info(s"Ingesting doc=${body.docId}")
        chunks  <- store.ingest(body.docId, body.text, body.metadata)
        resp    <- Ok(IngestResponse(body.docId, chunks.size).asJson)
      yield resp

    // -------------------------------------------------------------------------
    // POST /query  — RAG query → assembled context → (stub) LLM answer
    // -------------------------------------------------------------------------
    case req @ POST -> Root / "query" =>
      for
        body    <- req.as[QueryRequest]
        sid      = body.sessionId.getOrElse(UUID.randomUUID().toString)
        _       <- logger.info(s"Query sid=$sid q=${body.query}")
        window  <- freshWindow

        // Run the fs2 pipeline: retrieve → filter → dedup → assemble → inject
        _       <- ContextPipeline
                     .run(
                       rawChunks = store.queryStream(body.query, body.topK),
                       window    = window,
                       minScore  = 0.05,
                       k         = body.topK,
                     )
                     .compile.drain

        // Add the user query to the window
        _       <- window.addText(ContextRole.User, body.query, priority = 3.0)

        messages <- window.toMessages
        usedToks <- window.usedTokens

        // Stub answer — in production call Google ADK / Gemini here
        answer   = stubAnswer(body.query, messages)

        // Retrieve chunks for the response payload
        chunks  <- store.query(body.query, body.topK)
        resp    <- Ok(QueryResponse(answer, chunks, usedToks, sid).asJson)
      yield resp

    // -------------------------------------------------------------------------
    // GET /health
    // -------------------------------------------------------------------------
    case GET -> Root / "health" =>
      for
        n    <- store.count
        resp <- Ok(Map("status" -> "ok", "chunks" -> n.toString).asJson)
      yield resp
  }

  // ---------------------------------------------------------------------------
  // Stub answer — replace with real LLM call
  // ---------------------------------------------------------------------------
  private def stubAnswer(query: String, messages: List[Map[String, String]]): String =
    val contextBlocks = messages.filter(_("role") == "retrieval").map(_("content"))
    if contextBlocks.isEmpty then
      s"I don't have enough lore context to answer: '$query'"
    else
      s"Based on the lore context, here is what I know about '$query':\n\n" +
      contextBlocks.mkString("\n---\n")

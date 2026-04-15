package deeptutor

import cats.effect.IO
import cats.syntax.all.*
import io.circe.syntax.*
import org.http4s.*
import org.http4s.circe.*
import org.http4s.circe.CirceEntityDecoder.*
import org.http4s.dsl.io.*
import org.http4s.server.middleware.QueryParamDecoder
import org.typelevel.log4cats.Logger
import java.util.UUID
import fs2.Stream

// =============================================================================
// Http4s Routes — the HTTP face of the harness
// =============================================================================
// Context Engineering insight: the HTTP layer is *thin*.  It decodes requests,
// delegates to the pipeline, and encodes responses.  No business logic lives here.

class LoreRoutes(
    store: VectorStoreAlgebra[IO],
    costTracker: CostTrackerAlgebra[IO]   = null,
    traceStore:  TraceStoreAlgebra[IO]    = null,
)(using logger: Logger[IO]):

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
    // GET /health  — includes cache stats and cost summary
    // -------------------------------------------------------------------------
    case GET -> Root / "health" =>
      for
        n         <- store.count
        cacheStats = store match
                       case c: CachedVectorStore => Some(c.cacheStats)
                       case _                    => None
        cacheJson <- cacheStats.fold(IO.pure(Map.empty[String, String]))(_.map(s =>
                       Map("cache_hit_rate" -> f"${s.hitRate}%.3f", "cache_size" -> s.size.toString)
                     ))
        costJson  <- Option(costTracker).fold(IO.pure(Map.empty[String, String]))(t =>
                       t.summary.map(s => Map("total_cost_usd" -> f"${s.totalCostUsd}%.6f", "llm_calls" -> s.totalEvents.toString))
                     )
        resp      <- Ok((Map("status" -> "ok", "chunks" -> n.toString) ++ cacheJson ++ costJson).asJson)
      yield resp

    // -------------------------------------------------------------------------
    // GET /cost  — cost aggregation report
    // -------------------------------------------------------------------------
    case GET -> Root / "cost" =>
      Option(costTracker) match
        case None    => Ok(Map("error" -> "cost tracker not configured").asJson)
        case Some(t) => t.summary.flatMap(s => Ok(s.asJson))

    // -------------------------------------------------------------------------
    // GET /traces  — list recent turn IDs
    // -------------------------------------------------------------------------
    case GET -> Root / "traces" :? LimitParam(limit) =>
      Option(traceStore) match
        case None    => Ok(Map("turns" -> List.empty[TurnSummary]).asJson)
        case Some(ts) =>
          ts.listTurns(limit.getOrElse(20)).flatMap(turns =>
            Ok(Map("turns" -> turns).asJson)
          )

    // -------------------------------------------------------------------------
    // GET /trace/{turn_id}  — replay a single turn's events
    // -------------------------------------------------------------------------
    case GET -> Root / "trace" / turnId =>
      Option(traceStore) match
        case None     => NotFound(Map("error" -> s"trace store not configured").asJson)
        case Some(ts) =>
          ts.getTurn(turnId).flatMap { events =>
            if events.isEmpty then
              NotFound(Map("error" -> s"No trace for turn_id=$turnId").asJson)
            else
              val wallMs =
                if events.size > 1 then (events.last.ts - events.head.ts).toDouble else 0.0
              val errCount = events.count(_.kind == TraceEventKind.Error)
              Ok(Map(
                "turn_id"         -> turnId.asJson,
                "event_count"     -> events.size.asJson,
                "wall_latency_ms" -> wallMs.asJson,
                "error_count"     -> errCount.asJson,
                "events"          -> events.asJson,
              ).asJson)
          }
  }

  private object LimitParam extends OptionalQueryParamDecoderMatcher[Int]("limit")

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

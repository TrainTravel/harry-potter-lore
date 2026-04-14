package deeptutor

import cats.effect.IO
import cats.syntax.all.*
import fs2.{Pipe, Stream}

// =============================================================================
// Context Pipeline — fs2 streaming
// =============================================================================
// Context Engineering insight: modelling retrieval + assembly as a *stream*
// lets you:
//   - process chunks lazily (no OOM on large corpora)
//   - compose pipeline stages with type-safe Pipes
//   - add back-pressure naturally (fs2 handles it)
//   - stream partial results to the client (SSE / chunked HTTP)
//
// Pipeline stages:
//   queryStream
//     |> retrieveChunks        (Vector store → Stream[Chunk])
//     |> scoreFilter           (drop low-confidence chunks)
//     |> deduplicate           (Jaccard similarity)
//     |> mmrRerank             (Maximal Marginal Relevance)
//     |> assembleContext       (format into a context block)
//     |> injectIntoWindow      (add to ContextWindowAlgebra)

object ContextPipeline:

  // ---------------------------------------------------------------------------
  // Stage: filter by minimum score
  // ---------------------------------------------------------------------------

  def scoreFilter(minScore: Double): Pipe[IO, Chunk, Chunk] =
    _.filter(_.score >= minScore)

  // ---------------------------------------------------------------------------
  // Stage: deduplication using Jaccard similarity
  // ---------------------------------------------------------------------------

  def deduplicate(threshold: Double = 0.90): Pipe[IO, Chunk, Chunk] =
    stream =>
      stream.scan((List.empty[Chunk], Option.empty[Chunk])) { case ((seen, _), chunk) =>
        val isDup = seen.exists(s => jaccard(s.text, chunk.text) >= threshold)
        if isDup then (seen, None)
        else (seen :+ chunk, Some(chunk))
      }
      .collect { case (_, Some(chunk)) => chunk }

  // ---------------------------------------------------------------------------
  // Stage: take top-K by score
  // ---------------------------------------------------------------------------

  def topK(k: Int): Pipe[IO, Chunk, Chunk] =
    _.fold(List.empty[Chunk])(_ :+ _)
     .flatMap(chunks => Stream.emits(chunks.sortBy(-_.score).take(k)))

  // ---------------------------------------------------------------------------
  // Stage: assemble context block from chunks
  // ---------------------------------------------------------------------------

  def assembleContext(
      strategy: String = "citation",
      preamble: String = "Relevant Harry Potter lore:",
  ): Pipe[IO, Chunk, String] =
    _.fold(List.empty[Chunk])(_ :+ _)
     .map(chunks => format(chunks, strategy, preamble))

  // ---------------------------------------------------------------------------
  // Stage: inject assembled context into the window
  // ---------------------------------------------------------------------------

  def injectIntoWindow(window: ContextWindowAlgebra[IO]): Pipe[IO, String, ContextEntry] =
    _.evalMap { block =>
      window.addText(ContextRole.Retrieval, block, priority = 2.0)
    }

  // ---------------------------------------------------------------------------
  // Full pipeline composition
  // ---------------------------------------------------------------------------

  /** Run a full retrieval-to-injection pipeline from a stream of raw chunks. */
  def run(
      rawChunks: Stream[IO, Chunk],
      window: ContextWindowAlgebra[IO],
      minScore: Double = 0.1,
      k: Int = 5,
      strategy: String = "citation",
  ): Stream[IO, ContextEntry] =
    rawChunks
      .through(scoreFilter(minScore))
      .through(deduplicate(threshold = 0.90))
      .through(topK(k))
      .through(assembleContext(strategy))
      .through(injectIntoWindow(window))

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private def jaccard(a: String, b: String): Double =
    val sa = a.toLowerCase.split("\\s+").toSet
    val sb = b.toLowerCase.split("\\s+").toSet
    if sa.isEmpty && sb.isEmpty then 1.0
    else sa.intersect(sb).size.toDouble / sa.union(sb).size.toDouble

  private def format(chunks: List[Chunk], strategy: String, preamble: String): String =
    strategy match
      case "citation" =>
        val body = chunks.zipWithIndex.map { case (c, i) =>
          val src = c.metadata.getOrElse("doc_id", c.docId)
          s"[${i + 1}] $src — ${c.text}"
        }.mkString("\n\n")
        s"$preamble\n\n$body"

      case "sandwich" if chunks.nonEmpty =>
        val best = chunks.head
        val rest = chunks.tail.map(_.text).mkString("\n\n")
        s"$preamble\n\n[Key] ${best.text}\n\n$rest\n\n[Key repeated] ${best.text}"

      case _ =>
        val body = chunks.map(_.text).mkString("\n\n")
        s"$preamble\n\n$body"

package deeptutor

import cats.effect.IO
import cats.syntax.all.*
import io.circe.Encoder
import io.circe.generic.semiauto.*
import java.util.UUID

// =============================================================================
// Ingestion Pipeline DSL — deferred execution (Apache Beam / Spark style)
// =============================================================================
//
// Build a pure description of what to ingest, then hand it to the interpreter.
//
//   val plan = IngestionPipeline.empty
//     .add(Document[WikiArticle]("elon-musk",    wikiText,  "real_world", Some("2024-03")))
//     .add(Document[NewsArticle]("musk-twitter", newsText,  "real_world", Some("2022-10")))
//     .add(Document[FictionalCanon]("hp:horcruxes", loreText, "hp"))
//
//   // Nothing has run yet — `plan` is a plain value (List[PipelineStep])
//
//   plan.run(vectorStore)          // IO[List[IngestResult]]
//     .flatMap(results => ...)     // effects happen only here
//
// How typeclass dispatch works
// ────────────────────────────
// Document[S] is phantom-typed.  When you call `.add(doc)`, the compiler
// summons `Chunkable[S]` at the call site and captures `.chunk` and
// `.strategyName` into a `PipelineStep` (a plain case class with no type
// parameter).  The heterogeneous list `List[PipelineStep]` is therefore
// homogeneous at runtime — the type information was consumed at construction.
//
// This is the same trick circe uses: `Encoder[A]` is resolved at the call
// site, not inside the generic function that uses it.


// ---------------------------------------------------------------------------
// Document[S] — the DSL unit
// ---------------------------------------------------------------------------
// Phantom type S carries the source kind; it is erased at runtime but
// guides the compiler to the right Chunkable[S] instance.

final case class Document[S <: SourceKind](
    docId:    String,
    text:     String,
    universe: String,
    asOf:     Option[String] = None,
)


// ---------------------------------------------------------------------------
// PipelineStep — the resolved, type-erased description
// ---------------------------------------------------------------------------
// Once a Document[S] passes through `.add()`, S is gone.  What remains is
// a plain value the interpreter can work with uniformly.

final case class PipelineStep(
    docId:        String,
    text:         String,
    universe:     String,
    sourceKind:   String,          // e.g. "WikiArticle" — for metadata + tracing
    strategyName: String,          // e.g. "wiki"        — for metadata + tracing
    asOf:         Option[String],
    chunkFn:      String => List[String],   // captured from Chunkable[S]
) derives CanEqual:
  /** Apply the captured chunking function. Pure, no IO. */
  def chunks: List[String] = chunkFn(text)


// ---------------------------------------------------------------------------
// IngestResult — what the interpreter returns per step
// ---------------------------------------------------------------------------

final case class IngestResult(
    docId:       String,
    universe:    String,
    chunkCount:  Int,
    strategy:    String,
) derives Encoder.AsObject


// ---------------------------------------------------------------------------
// IngestionPipeline — the pure plan
// ---------------------------------------------------------------------------

final class IngestionPipeline private (val steps: List[PipelineStep]):

  /**
   * Add a document to the plan.
   *
   * The compiler summons `Chunkable[S]` here and captures its functions.
   * After this call the type parameter S is erased — the pipeline stores
   * only `PipelineStep` values.
   *
   *   pipeline.add(Document[WikiArticle]("elon-musk", text, "real_world"))
   *   //            ^^^^^^^^^^^^ compiler looks up Chunkable[WikiArticle]
   */
  def add[S <: SourceKind: Chunkable](doc: Document[S]): IngestionPipeline =
    val c    = Chunkable[S]
    val step = PipelineStep(
      docId        = doc.docId,
      text         = doc.text,
      universe     = doc.universe,
      sourceKind   = doc.getClass.getSimpleName.stripSuffix("$"),
      strategyName = c.strategyName,
      asOf         = doc.asOf,
      chunkFn      = c.chunk,
    )
    new IngestionPipeline(steps :+ step)

  /** Number of documents in the plan (not chunks). */
  def size: Int = steps.size

  /** Human-readable summary of the plan — inspect before running. */
  def describe: String =
    val header = s"IngestionPipeline: ${size} document(s)"
    val rows   = steps.zipWithIndex.map { case (s, i) =>
      val asOf = s.asOf.map(d => s" as_of=$d").getOrElse("")
      f"  [${i + 1}%02d] ${s.docId}%-30s universe=${s.universe}%-12s strategy=${s.strategyName}$asOf"
    }
    (header +: rows).mkString("\n")

  /**
   * Run the pipeline against a VectorStoreAlgebra.
   *
   * This is the IO boundary — the single place where effects happen.
   * All chunking (pure) runs inside this call, then chunks are stored.
   *
   * Sequential by default; switch to `.parTraverse` for parallel ingestion
   * when the vector store supports concurrent writes.
   */
  /**
   * Run the pipeline sequentially.
   *
   * Chunking (pure) runs first for each step, then the resulting `Chunk`
   * objects are upserted in bulk.  The vector store is never asked to
   * re-chunk — it receives pre-split chunks produced by the typeclass.
   */
  def run(store: VectorStoreAlgebra[IO]): IO[List[IngestResult]] =
    steps.traverse(runStep(store))

  /**
   * Run the pipeline with parallel IO.
   *
   * Each document's upsert runs concurrently.  Chunking is still pure and
   * sequential (cheap); only the IO boundary is parallelised.
   */
  def runParallel(store: VectorStoreAlgebra[IO]): IO[List[IngestResult]] =
    steps.parTraverse(runStep(store))

  private def runStep(store: VectorStoreAlgebra[IO])(step: PipelineStep): IO[IngestResult] =
    val chunkObjects = step.chunks.zipWithIndex.map { case (text, i) =>
      // Construct Chunk directly — the pipeline owns chunking, the store owns storage.
      Chunk(
        chunkId  = UUID.randomUUID().toString,
        docId    = step.docId,
        text     = text,
        score    = 0.0,
        metadata = Map(
          "universe"    -> step.universe,
          "source_kind" -> step.sourceKind,
          "strategy"    -> step.strategyName,
          "as_of"       -> step.asOf.getOrElse(""),
          "chunk_index" -> i.toString,
        ),
      )
    }
    store.upsert(chunkObjects).as(
      IngestResult(step.docId, step.universe, chunkObjects.size, step.strategyName)
    )


object IngestionPipeline:
  val empty: IngestionPipeline = new IngestionPipeline(List.empty)

  /**
   * Convenience: build from a sequence of (Document[S], Chunkable[S]) pairs.
   *
   * Because Scala can't directly express heterogeneous lists in a varargs
   * position, the idiomatic builder is `.add()` chaining.  This factory
   * is useful for building a pipeline from a homogeneous list of one type:
   *
   *   IngestionPipeline.fromWiki(wikiDocs*)
   */
  def fromWiki(docs: Document[WikiArticle]*): IngestionPipeline =
    docs.foldLeft(empty)(_.add(_))

  def fromFictionalCanon(docs: Document[FictionalCanon]*): IngestionPipeline =
    docs.foldLeft(empty)(_.add(_))

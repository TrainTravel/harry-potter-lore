package deeptutor

import cats.effect.{IO, Ref}
import cats.syntax.all.*
import fs2.Stream
import java.util.UUID

// =============================================================================
// VectorStore Algebra + In-Memory Interpreter
// =============================================================================
// In production this would call a ChromaDB HTTP API (or a gRPC client).
// The algebra lets us swap the backend without touching pipeline or server code.

trait VectorStoreAlgebra[F[_]]:
  /** Upsert a batch of chunks into the store. */
  def upsert(chunks: List[Chunk]): F[Unit]

  /** Ingest raw text: chunk it, then upsert. */
  def ingest(docId: String, text: String, metadata: Map[String, String]): F[List[Chunk]]

  /** Retrieve the top-k most relevant chunks for a query. */
  def query(queryText: String, topK: Int): F[List[Chunk]]

  /** Stream chunks for a query (useful for large k). */
  def queryStream(queryText: String, topK: Int): Stream[F, Chunk]

  /** Total number of chunks stored. */
  def count: F[Int]


object VectorStoreAlgebra:

  // ---------------------------------------------------------------------------
  // In-memory interpreter — keyword overlap scoring (no GPU required)
  // ---------------------------------------------------------------------------

  def inMemory: IO[VectorStoreAlgebra[IO]] =
    Ref.of[IO, Map[String, Chunk]](Map.empty).map { store =>
      new VectorStoreAlgebra[IO]:

        def upsert(chunks: List[Chunk]): IO[Unit] =
          store.update(m => m ++ chunks.map(c => c.chunkId -> c))

        def ingest(docId: String, text: String, metadata: Map[String, String]): IO[List[Chunk]] =
          val paragraphs = text.split("\n\n+").filter(_.trim.nonEmpty)
          val chunks = paragraphs.zipWithIndex.map { case (para, i) =>
            Chunk(
              chunkId  = UUID.randomUUID().toString,
              docId    = docId,
              text     = para.trim,
              score    = 0.0,
              metadata = metadata + ("chunk_index" -> i.toString),
            )
          }.toList
          upsert(chunks).as(chunks)

        def query(queryText: String, topK: Int): IO[List[Chunk]] =
          store.get.map { m =>
            val queryWords = queryText.toLowerCase.split("\\s+").toSet
            m.values.toList
              .map { chunk =>
                val words   = chunk.text.toLowerCase.split("\\s+").toSet
                val overlap = (queryWords & words).size.toDouble / (queryWords | words).size.toDouble
                chunk.copy(score = overlap)
              }
              .sortBy(-_.score)
              .take(topK)
          }

        def queryStream(queryText: String, topK: Int): Stream[IO, Chunk] =
          Stream.eval(query(queryText, topK)).flatMap(Stream.emits)

        def count: IO[Int] =
          store.get.map(_.size)
    }

  // ---------------------------------------------------------------------------
  // ChromaDB HTTP interpreter stub
  // ---------------------------------------------------------------------------
  // Replace `inMemory` with this in production.
  // Requires the Python ChromaDB server running at http://localhost:8000.

  def chromadb(baseUrl: String, collectionName: String): IO[VectorStoreAlgebra[IO]] =
    // TODO: implement using http4s EmberClient + Circe
    // For now, delegate to inMemory to keep compilation green.
    IO.println(s"[VectorStore] ChromaDB at $baseUrl (stub — using in-memory fallback)") *>
    inMemory

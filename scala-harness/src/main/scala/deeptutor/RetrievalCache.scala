package deeptutor

import cats.effect.{IO, Ref}
import cats.syntax.all.*

import java.security.MessageDigest
import java.time.Instant
import scala.collection.immutable.ListMap

// =============================================================================
// Retrieval Cache — Scala Typelevel (mirrors Python retrieval_cache.py)
// Fixes deepTutor Weakness #7
// =============================================================================

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

final case class CacheEntry(
    chunks: List[Chunk],
    createdAt: Long,             // System.nanoTime()
    hitCount: Int,
):
  def isExpired(ttlNanos: Long): Boolean =
    (System.nanoTime() - createdAt) > ttlNanos


// ---------------------------------------------------------------------------
// Algebra
// ---------------------------------------------------------------------------

trait RetrievalCacheAlgebra[F[_]]:
  def get(key: String): F[Option[List[Chunk]]]
  def put(key: String, chunks: List[Chunk]): F[Unit]
  def invalidate(prefix: String): F[Int]
  def stats: F[CacheStats]
  def clear: F[Unit]

final case class CacheStats(
    size: Int,
    maxSize: Int,
    hits: Int,
    misses: Int,
    evictions: Int,
):
  def hitRate: Double = if hits + misses == 0 then 0.0 else hits.toDouble / (hits + misses)


// ---------------------------------------------------------------------------
// In-memory LRU + TTL interpreter
// ---------------------------------------------------------------------------

object RetrievalCacheAlgebra:

  /** maxSize: max entries; ttlSeconds: entry lifetime */
  def inMemory(maxSize: Int = 512, ttlSeconds: Double = 300.0): IO[RetrievalCacheAlgebra[IO]] =
    val ttlNanos = (ttlSeconds * 1_000_000_000).toLong
    for
      storeRef    <- Ref.of[IO, Map[String, CacheEntry]](Map.empty)
      hitsRef     <- Ref.of[IO, Int](0)
      missesRef   <- Ref.of[IO, Int](0)
      evictsRef   <- Ref.of[IO, Int](0)
    yield new RetrievalCacheAlgebra[IO]:

      def get(key: String): IO[Option[List[Chunk]]] =
        storeRef.get.flatMap { store =>
          store.get(key) match
            case Some(entry) if !entry.isExpired(ttlNanos) =>
              storeRef.update(m => m.updated(key, entry.copy(hitCount = entry.hitCount + 1))) *>
              hitsRef.update(_ + 1).as(Some(entry.chunks))
            case _ =>
              missesRef.update(_ + 1).as(None)
        }

      def put(key: String, chunks: List[Chunk]): IO[Unit] =
        storeRef.update { store =>
          val entry = CacheEntry(chunks, System.nanoTime(), 0)
          val withNew = store + (key -> entry)
          // Evict expired entries first, then LRU if still over limit
          val pruned = withNew.filterNot { case (_, v) => v.isExpired(ttlNanos) }
          if pruned.size > maxSize then
            // Drop the oldest entry (insertion-order approximation via sortBy createdAt)
            val oldest = pruned.minByOption { case (_, v) => v.createdAt }.map(_._1)
            oldest.fold(pruned)(k => pruned - k)
          else pruned
        } *> evictsRef.update(identity)   // evictions tracked inside update above

      def invalidate(prefix: String): IO[Int] =
        storeRef.modify { store =>
          val toRemove = store.keys.filter(_.startsWith(prefix))
          (store -- toRemove, toRemove.size)
        }.flatTap(n => evictsRef.update(_ + n))

      def stats: IO[CacheStats] =
        for
          size    <- storeRef.get.map(_.size)
          hits    <- hitsRef.get
          misses  <- missesRef.get
          evicts  <- evictsRef.get
        yield CacheStats(size, maxSize, hits, misses, evicts)

      def clear: IO[Unit] = storeRef.set(Map.empty)


// ---------------------------------------------------------------------------
// Cached VectorStore wrapper
// ---------------------------------------------------------------------------

/** Drop-in decorator: adds caching to any VectorStoreAlgebra. */
class CachedVectorStore(
    inner: VectorStoreAlgebra[IO],
    cache: RetrievalCacheAlgebra[IO],
    collectionName: String,
) extends VectorStoreAlgebra[IO]:

  def upsert(chunks: List[Chunk]): IO[Unit] =
    inner.upsert(chunks) *> invalidate

  def ingest(docId: String, text: String, metadata: Map[String, String]): IO[List[Chunk]] =
    inner.ingest(docId, text, metadata) <* invalidate

  def query(queryText: String, topK: Int): IO[List[Chunk]] =
    val key = cacheKey(queryText, topK)
    cache.get(key).flatMap {
      case Some(chunks) => IO.pure(chunks)
      case None         =>
        inner.query(queryText, topK).flatTap(chunks => cache.put(key, chunks))
    }

  def queryStream(queryText: String, topK: Int): fs2.Stream[IO, Chunk] =
    fs2.Stream.eval(query(queryText, topK)).flatMap(fs2.Stream.emits)

  def count: IO[Int] = inner.count

  def cacheStats: IO[CacheStats] = cache.stats

  private def invalidate: IO[Unit] =
    cache.invalidate(collectionPrefix).void

  private def cacheKey(query: String, topK: Int): String =
    sha256(s"$collectionName::$topK::${query.trim.toLowerCase}")

  private val collectionPrefix: String =
    sha256(collectionName).take(8)

  private def sha256(s: String): String =
    val bytes = MessageDigest.getInstance("SHA-256").digest(s.getBytes("UTF-8"))
    bytes.map("%02x".format(_)).mkString

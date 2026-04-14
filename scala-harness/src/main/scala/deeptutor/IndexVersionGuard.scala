package deeptutor

import cats.effect.IO
import io.circe.{Encoder, Decoder}
import io.circe.generic.semiauto.*
import io.circe.parser.*
import io.circe.syntax.*

import java.nio.file.{Files, Path, Paths, StandardCopyOption}
import java.time.Instant

// =============================================================================
// Index Version Guard — Scala Typelevel (mirrors Python index_version_guard.py)
// Fixes deepTutor Weakness #6
// =============================================================================

final case class IndexManifest(
    schemaVersion: Int,
    embeddingModel: String,
    embeddingDim: Int,
    collectionName: String,
    createdAt: Long,
    updatedAt: Long,
    docCount: Int,
    chunkCount: Int,
) derives Encoder.AsObject, Decoder

object IndexManifest:
  val CurrentSchemaVersion: Int = 1

  def create(model: String, dim: Int, collection: String): IndexManifest =
    val now = Instant.now().toEpochMilli
    IndexManifest(CurrentSchemaVersion, model, dim, collection, now, now, 0, 0)


// ---------------------------------------------------------------------------
// Typed errors
// ---------------------------------------------------------------------------

case class IndexVersionMismatch(stored: String, current: String) extends DomainError:
  val message = s"Embedding model mismatch — index: $stored, current: $current. Run rebuild()."

case class IndexDimensionMismatch(stored: Int, current: Int) extends DomainError:
  val message = s"Embedding dimension mismatch — index: $stored, current: $current."

case class ManifestSchemaTooOld(stored: Int, current: Int) extends DomainError:
  val message = s"Manifest schema v$stored < current v$current. Run migrate()."


// ---------------------------------------------------------------------------
// Guard algebra
// ---------------------------------------------------------------------------

trait IndexVersionGuardAlgebra[F[_]]:
  /** Initialise if no manifest; validate if one exists. Returns the manifest. */
  def ensure: F[IndexManifest]

  /** Validate only. Raises a typed error on mismatch. */
  def validate: F[IndexManifest]

  /** Update doc/chunk counts after ingestion. */
  def updateStats(docCount: Int, chunkCount: Int): F[Unit]


// ---------------------------------------------------------------------------
// File-system interpreter
// ---------------------------------------------------------------------------

object IndexVersionGuardAlgebra:

  def fileSystem(
      manifestPath: String,
      embeddingModel: String,
      embeddingDim: Int,
      collectionName: String,
  ): IndexVersionGuardAlgebra[IO] =
    val path = Paths.get(manifestPath)

    new IndexVersionGuardAlgebra[IO]:

      def ensure: IO[IndexManifest] =
        IO(Files.exists(path)).flatMap {
          case false => create
          case true  => validate
        }

      def validate: IO[IndexManifest] =
        for
          raw      <- IO(Files.readString(path))
          manifest <- IO.fromEither(decode[IndexManifest](raw).left.map(e => new RuntimeException(e.message)))
          _        <- checkSchema(manifest)
          _        <- checkModel(manifest)
          _        <- checkDim(manifest)
          _        <- checkCollection(manifest)
        yield manifest

      def updateStats(docCount: Int, chunkCount: Int): IO[Unit] =
        for
          raw      <- IO(Files.readString(path))
          manifest <- IO.fromEither(decode[IndexManifest](raw).left.map(e => new RuntimeException(e.message)))
          updated   = manifest.copy(
                        docCount   = docCount,
                        chunkCount = chunkCount,
                        updatedAt  = Instant.now().toEpochMilli,
                      )
          _        <- write(updated)
        yield ()

      // -- Helpers -----------------------------------------------------------

      private def create: IO[IndexManifest] =
        val m = IndexManifest.create(embeddingModel, embeddingDim, collectionName)
        write(m).as(m)

      private def write(m: IndexManifest): IO[Unit] =
        IO {
          val parent = path.getParent
          if parent != null then Files.createDirectories(parent)
          Files.writeString(path, m.asJson.spaces2)
        }

      private def checkSchema(m: IndexManifest): IO[Unit] =
        if m.schemaVersion < IndexManifest.CurrentSchemaVersion then
          IO.raiseError(ManifestSchemaTooOld(m.schemaVersion, IndexManifest.CurrentSchemaVersion))
        else IO.unit

      private def checkModel(m: IndexManifest): IO[Unit] =
        if m.embeddingModel != embeddingModel then
          IO.raiseError(IndexVersionMismatch(m.embeddingModel, embeddingModel))
        else IO.unit

      private def checkDim(m: IndexManifest): IO[Unit] =
        if m.embeddingDim != embeddingDim then
          IO.raiseError(IndexDimensionMismatch(m.embeddingDim, embeddingDim))
        else IO.unit

      private def checkCollection(m: IndexManifest): IO[Unit] =
        if m.collectionName != collectionName then
          IO.raiseError(IndexVersionMismatch(m.collectionName, collectionName))
        else IO.unit

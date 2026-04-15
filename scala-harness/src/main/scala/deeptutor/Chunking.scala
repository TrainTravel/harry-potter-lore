package deeptutor

// =============================================================================
// Chunking typeclass + DSL
// =============================================================================
//
// Design: typeclass dispatch (same pattern as circe's Encoder[A] / Decoder[A]).
//
// Every source kind has a `given Chunkable[S]` instance.  When you write
//
//   Document[Wiki]("elon-musk", text, "real_world")
//
// the compiler summons `Chunkable[Wiki]` and captures its `chunk` function
// into the resulting `PipelineStep`.  No runtime match, no dictionary lookup —
// the strategy is resolved statically at the call site.
//
// Algorithms live in `ChunkingAlgorithms` (pure String => List[String]).
// The typeclass instances simply delegate to the right algorithm.
// Adding a new source kind = add a phantom type + one `given` line.
//
// ---------------------------------------------------------------------------
// Phantom types — one per source kind
// ---------------------------------------------------------------------------
// These are never instantiated; they exist only as type parameters so the
// compiler can select the right Chunkable instance at compile time.

sealed trait SourceKind
sealed trait FictionalCanon extends SourceKind   // HP, LotR, Witcher, …
sealed trait WikiArticle    extends SourceKind   // Wikipedia / wiki-style articles
sealed trait Biography      extends SourceKind   // long-form biographical text
sealed trait NewsArticle    extends SourceKind   // time-sensitive news content


// ---------------------------------------------------------------------------
// Chunkable typeclass
// ---------------------------------------------------------------------------

trait Chunkable[S <: SourceKind]:
  /** Splits `text` into a list of non-empty chunk strings. */
  def chunk(text: String): List[String]

  /** Short name stored in chunk metadata and trace logs. */
  def strategyName: String


object Chunkable:
  /** Summoner — `Chunkable[Wiki]` instead of `summon[Chunkable[Wiki]]`. */
  def apply[S <: SourceKind](using c: Chunkable[S]): Chunkable[S] = c

  // ── Given instances ─────────────────────────────────────────────────────
  //
  // Each instance maps one source kind to one algorithm.
  // The mapping is justified in the comment on each instance.

  /** Narrative lore: paragraph-aware with word-count fallback. */
  given Chunkable[FictionalCanon] with
    val strategyName = "recursive"
    def chunk(text: String): List[String] = ChunkingAlgorithms.recursive(text)

  /** Wikipedia / wiki articles: respect ## section header boundaries. */
  given Chunkable[WikiArticle] with
    val strategyName = "wiki"
    def chunk(text: String): List[String] = ChunkingAlgorithms.wiki(text)

  /** Biographical text: same as fictional (narrative structure, long-form). */
  given Chunkable[Biography] with
    val strategyName = "recursive"
    def chunk(text: String): List[String] = ChunkingAlgorithms.recursive(text)

  /** News articles: paragraph-per-chunk preserves the inverted pyramid. */
  given Chunkable[NewsArticle] with
    val strategyName = "paragraph"
    def chunk(text: String): List[String] = ChunkingAlgorithms.paragraph(text)


// ---------------------------------------------------------------------------
// Chunking algorithms — pure functions, no effects
// ---------------------------------------------------------------------------

object ChunkingAlgorithms:

  private val wordLimit = 300
  private val overlap   = 50

  /** Fixed sliding window over words with overlap. */
  def fixed(text: String, size: Int = wordLimit, ov: Int = overlap): List[String] =
    val words = text.split("\\s+").toList
    if words.isEmpty then List.empty
    else
      LazyList
        .iterate(0)(_ + (size - ov))
        .takeWhile(_ < words.size)
        .map(start => words.slice(start, start + size).mkString(" "))
        .toList

  /** Split on sentence boundaries, grouping `n` sentences per chunk. */
  def sentence(text: String, n: Int = 5): List[String] =
    val sentences = text.trim.split("(?<=[.!?])\\s+").toList
    sentences.grouped(n).map(_.mkString(" ")).toList

  /** Split on blank lines. */
  def paragraph(text: String): List[String] =
    text.split("\\n\\s*\\n").map(_.trim).filter(_.nonEmpty).toList

  /**
   * Paragraph → word-count fallback.
   * Keeps semantic paragraph boundaries when paragraphs are short enough.
   * Falls back to fixed chunks for very long paragraphs.
   */
  def recursive(text: String, size: Int = wordLimit): List[String] =
    paragraph(text).flatMap { para =>
      if para.split("\\s+").length <= size then List(para)
      else fixed(para, size)
    }

  /**
   * Section-header aware chunking for Wikipedia / markdown documents.
   *
   * Splits just before lines starting with one or more `#` characters so each
   * section becomes its own chunk.  The header is kept as the first line of
   * the chunk, anchoring the section topic inside the embedding.
   *
   * Oversized sections (> 400 words) are sub-chunked with `recursive`.
   *
   * Example:
   *   "## Early life\nBorn in..."  →  chunk: "## Early life\nBorn in..."
   *   "## Career\nFounded..."      →  chunk: "## Career\nFounded..."
   */
  def wiki(text: String): List[String] =
    val sections = text.split("(?=\\n#{1,3}\\s)").map(_.trim).filter(_.nonEmpty).toList
    val result = sections.flatMap { sec =>
      if sec.split("\\s+").length > 400 then recursive(sec)
      else List(sec)
    }
    if result.nonEmpty then result else paragraph(text)

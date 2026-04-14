package deeptutor

import cats.effect.IO
import fs2.Stream
import munit.CatsEffectSuite

class ContextPipelineSuite extends CatsEffectSuite:

  private def makeChunk(id: String, text: String, score: Double): Chunk =
    Chunk(chunkId = id, docId = "test", text = text, score = score)

  test("scoreFilter removes chunks below threshold"):
    val chunks = Stream.emits[IO, Chunk](List(
      makeChunk("a", "relevant text about Dumbledore", 0.8),
      makeChunk("b", "low score snippet", 0.02),
      makeChunk("c", "moderate text about Hogwarts", 0.5),
    ))

    chunks.through(ContextPipeline.scoreFilter(0.1)).compile.toList.map { result =>
      assertEquals(result.size, 2)
      assert(result.forall(_.score >= 0.1))
    }

  test("deduplicate drops near-identical chunks"):
    val chunks = Stream.emits[IO, Chunk](List(
      makeChunk("a", "Dumbledore is the headmaster of Hogwarts", 0.9),
      makeChunk("b", "Dumbledore is the headmaster of Hogwarts", 0.85), // duplicate
      makeChunk("c", "The Chamber of Secrets is below Hogwarts", 0.7),
    ))

    chunks.through(ContextPipeline.deduplicate(threshold = 0.85)).compile.toList.map { result =>
      assertEquals(result.size, 2, "near-duplicate should be removed")
    }

  test("topK limits output to k chunks"):
    val chunks = Stream.emits[IO, Chunk](
      (1 to 10).map(i => makeChunk(s"c$i", s"chunk $i content words", i.toDouble / 10)).toList
    )

    chunks.through(ContextPipeline.topK(3)).compile.toList.map { result =>
      assertEquals(result.size, 3)
    }

  test("full pipeline injects context into window"):
    for
      window  <- ContextWindowAlgebra.inMemory(maxTokens = 4096)
      store   <- VectorStoreAlgebra.inMemory
      _       <- store.ingest("hp1", "Dumbledore is the greatest wizard of his age.", Map.empty)
      _       <- store.ingest("hp2", "Harry Potter defeated Lord Voldemort in the Battle of Hogwarts.", Map.empty)
      _       <- ContextPipeline.run(
                   rawChunks = store.queryStream("Dumbledore wizard", 5),
                   window    = window,
                   minScore  = 0.0,
                   k         = 5,
                 ).compile.drain
      entries <- window.entries
    yield
      assert(entries.exists(_.role == ContextRole.Retrieval), "pipeline should inject retrieval entry")

package deeptutor

import cats.effect.IO
import munit.CatsEffectSuite

// =============================================================================
// Unit tests for ContextWindowAlgebra (in-memory interpreter)
// Uses munit-cats-effect for IO-aware assertions.
// =============================================================================

class ContextWindowSuite extends CatsEffectSuite:

  test("adds entries and tracks token usage"):
    for
      window <- ContextWindowAlgebra.inMemory(maxTokens = 1000, reservedOutput = 100)
      _      <- window.addText(ContextRole.System, "You are a lore expert.", priority = 10.0)
      _      <- window.addText(ContextRole.User, "Who is Dumbledore?", priority = 1.0)
      used   <- window.usedTokens
      es     <- window.entries
    yield
      assertEquals(es.size, 2)
      assert(used > 0, "used tokens should be positive")

  test("evicts lowest-priority non-system entry when budget exceeded"):
    // budget = 50 tokens
    for
      window <- ContextWindowAlgebra.inMemory(maxTokens = 60, reservedOutput = 10)
      // system entries are pinned (never evicted)
      _      <- window.addText(ContextRole.System, "System prompt", priority = 10.0)
      // low-priority user entry (will be evicted)
      _      <- window.addText(ContextRole.User, "Old question about quidditch teams", priority = 0.1)
      // high-priority retrieval entry (forces eviction of above)
      _      <- window.addText(ContextRole.Retrieval, "Dumbledore is the headmaster of Hogwarts School and the greatest wizard", priority = 2.0)
      es     <- window.entries
    yield
      // the low-priority user entry should have been evicted
      assert(!es.exists(_.role == ContextRole.User), "low-priority user entry should be evicted")
      assert(es.exists(_.role == ContextRole.System), "system entry must survive eviction")

  test("clearRetrieval removes only retrieval entries"):
    for
      window <- ContextWindowAlgebra.inMemory()
      _      <- window.addText(ContextRole.System, "System prompt.", priority = 10.0)
      _      <- window.addText(ContextRole.Retrieval, "Lore chunk 1", priority = 2.0)
      _      <- window.addText(ContextRole.User, "Question", priority = 1.0)
      _      <- window.clearRetrieval
      es     <- window.entries
    yield
      assertEquals(es.count(_.role == ContextRole.Retrieval), 0)
      assertEquals(es.count(_.role == ContextRole.System), 1)
      assertEquals(es.count(_.role == ContextRole.User), 1)

  test("toMessages renders role as lowercase string"):
    for
      window <- ContextWindowAlgebra.inMemory()
      _      <- window.addText(ContextRole.User, "Hello", priority = 1.0)
      msgs   <- window.toMessages
    yield
      assertEquals(msgs.head("role"), "user")
      assertEquals(msgs.head("content"), "Hello")

  test("reset clears all entries"):
    for
      window <- ContextWindowAlgebra.inMemory()
      _      <- window.addText(ContextRole.User, "A", priority = 1.0)
      _      <- window.reset
      es     <- window.entries
    yield assertEquals(es.size, 0)

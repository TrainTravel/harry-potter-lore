package deeptutor

import cats.effect.IO
import munit.CatsEffectSuite

// =============================================================================
// End-to-end tests for OpenAnalysisAgent + PerspectiveShiftAgent.
//
// These are *example* tests — property tests live in the per-algebra suites.
// The invariants pinned here are the compositional ones (retrieval fallback,
// prompt content, error paths).
// =============================================================================

class AgentSuite extends CatsEffectSuite:

  // ---- helpers ------------------------------------------------------------

  private def fakeResponse(
      analysis: String = "An analysis",
      corpusFacts: String = "Some corpus facts",
      ownReasoning: String = "Own reasoning here",
      citations: String = "[doc-a]",
  ): String =
    s"""[[ ## analysis ## ]]
       |$analysis
       |
       |[[ ## corpus_facts ## ]]
       |$corpusFacts
       |
       |[[ ## own_reasoning ## ]]
       |$ownReasoning
       |
       |[[ ## citations ## ]]
       |$citations""".stripMargin

  private def perspectiveResponse(
      principle: String = "Core principle",
      insight:   String = "Applied insight",
      reasoning: String = "Bridge",
      response:  String = "Voice speaking",
      citations: String = "[doc-x]",
  ): String =
    s"""[[ ## character_principle ## ]]
       |$principle
       |
       |[[ ## applied_insight ## ]]
       |$insight
       |
       |[[ ## reasoning ## ]]
       |$reasoning
       |
       |[[ ## character_response ## ]]
       |$response
       |
       |[[ ## citations ## ]]
       |$citations""".stripMargin

  // -------------------------------------------------------------------------
  // OpenAnalysisAgent
  // -------------------------------------------------------------------------

  test("OpenAnalysisAgent: happy path returns parsed analysis and citations"):
    for
      store <- VectorStoreAlgebra.inMemory
      _     <- store.ingest("hp1", "Dumbledore was headmaster of Hogwarts",
                            Map.empty)
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      llm   <- FakeLLMClient.of[IO](fakeResponse(
                 analysis     = "Dumbledore's complexity is a study in moral ambiguity.",
                 corpusFacts  = "Headmaster of Hogwarts. [hp1]",
                 ownReasoning = "Moral ambiguity is a Romantic trope.",
                 citations    = "[hp1]",
               ))
      result <- OpenAnalysisAgent
                  .make[IO](store, llm, conv)
                  .ask(
                    question       = "Why is Dumbledore morally complex?",
                    conversationId = None,
                  )
    yield
      result match
        case Right(out) =>
          assertEquals(out.fields.get("analysis"),
            Some("Dumbledore's complexity is a study in moral ambiguity."))
          assert(out.citations.contains("[hp1]"))
        case Left(err) =>
          fail(s"expected Right, got Left($err)")

  test("OpenAnalysisAgent: passes retrieved context into prompt"):
    for
      store <- VectorStoreAlgebra.inMemory
      _     <- store.ingest("hp1", "UNIQUE-CONTEXT-SIGIL about Dumbledore",
                            Map.empty)
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      llm   <- FakeLLMClient.of[IO](fakeResponse())
      _     <- OpenAnalysisAgent.make[IO](store, llm, conv)
                 .ask("Tell me about Dumbledore", None)
      prompts <- llm.recordedPrompts
    yield
      assert(prompts.nonEmpty)
      assert(prompts.head.prompt.contains("UNIQUE-CONTEXT-SIGIL"),
        s"prompt did not include retrieved context:\n${prompts.head.prompt}")

  test("OpenAnalysisAgent: multi-turn injects formatted chat_history"):
    for
      store <- VectorStoreAlgebra.inMemory
      _     <- store.ingest("hp1", "Dumbledore was headmaster", Map.empty)
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      cid    = ConversationId("conv-A")
      // First turn: stored in conv store directly
      _     <- conv.save(cid, Turn(
                 userMessage   = "previous question",
                 agentResponse = Map("analysis" -> "PRIOR-ANALYSIS-SIGIL"),
                 mode          = Mode.OpenAnalysis,
               ))
      llm   <- FakeLLMClient.of[IO](fakeResponse())
      _     <- OpenAnalysisAgent.make[IO](store, llm, conv)
                 .ask("follow-up question", Some(cid))
      prompts <- llm.recordedPrompts
    yield
      assert(prompts.head.prompt.contains("PRIOR-ANALYSIS-SIGIL"),
        s"chat_history not injected; prompt:\n${prompts.head.prompt}")

  test("OpenAnalysisAgent: multi-turn persists the new turn"):
    for
      store <- VectorStoreAlgebra.inMemory
      _     <- store.ingest("hp1", "Some context", Map.empty)
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      cid    = ConversationId("conv-B")
      llm   <- FakeLLMClient.of[IO](fakeResponse(analysis = "NEW-ANALYSIS"))
      _     <- OpenAnalysisAgent.make[IO](store, llm, conv).ask("q1", Some(cid))
      saved <- conv.load(cid, 10)
    yield
      assertEquals(saved.size, 1)
      assertEquals(saved.head.userMessage, "q1")
      assertEquals(saved.head.agentResponse.get("analysis"), Some("NEW-ANALYSIS"))
      assertEquals(saved.head.mode, Mode.OpenAnalysis)

  // -------------------------------------------------------------------------
  // PerspectiveShiftAgent
  // -------------------------------------------------------------------------

  test("PerspectiveShiftAgent: happy path returns character_response"):
    for
      store <- VectorStoreAlgebra.inMemory
      _     <- store.ingest("dumb1", "Albus Dumbledore on love and sacrifice",
                            Map("character" -> "albus-dumbledore"))
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      llm   <- FakeLLMClient.of[IO](perspectiveResponse(
                 principle = "Love is the oldest magic.",
                 response  = "My dear, think of this...",
               ))
      result <- PerspectiveShiftAgent.make[IO](store, llm, conv)
                  .ask(
                    scenario       = "I'm burned out at work",
                    character      = "Dumbledore",
                    conversationId = None,
                  )
    yield
      result match
        case Right(out) =>
          assertEquals(out.fields.get("character_response"),
            Some("My dear, think of this..."))
        case Left(err) =>
          fail(s"expected Right, got Left($err)")

  test("PerspectiveShiftAgent: character-scoped retrieval hits the filtered chunk"):
    for
      store <- VectorStoreAlgebra.inMemory
      _     <- store.ingest("snape1", "SNAPE-SCOPED-SIGIL loyalty sacrifice",
                            Map("character" -> "severus-snape"))
      _     <- store.ingest("other", "generic courage bravery",
                            Map("character" -> "harry-potter"))
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      llm   <- FakeLLMClient.of[IO](perspectiveResponse())
      _     <- PerspectiveShiftAgent.make[IO](store, llm, conv)
                 .ask("staying loyal in hard times", "Snape", None)
      prompts <- llm.recordedPrompts
    yield
      assert(prompts.head.prompt.contains("SNAPE-SCOPED-SIGIL"),
        s"character-scoped retrieval didn't find snape chunk; prompt:\n${prompts.head.prompt}")
      assert(!prompts.head.prompt.contains("generic courage bravery"),
        s"unfiltered chunk leaked into character-scoped prompt:\n${prompts.head.prompt}")

  test("PerspectiveShiftAgent: falls back to generic retrieval when no character-filtered chunks"):
    for
      store <- VectorStoreAlgebra.inMemory
      // No chunk tagged with character metadata
      _     <- store.ingest("g1", "GENERIC-FALLBACK-SIGIL on leadership",
                            Map.empty)
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      llm   <- FakeLLMClient.of[IO](perspectiveResponse())
      _     <- PerspectiveShiftAgent.make[IO](store, llm, conv)
                 .ask("becoming a leader", "UnknownCharacter", None)
      prompts <- llm.recordedPrompts
    yield
      assert(prompts.head.prompt.contains("GENERIC-FALLBACK-SIGIL"),
        s"fallback retrieval failed; prompt:\n${prompts.head.prompt}")

  test("PerspectiveShiftAgent: character_response header appears LAST in rendered prompt"):
    // This pins the voice-last pattern: if analysis fields are declared before
    // character_response, the rendered prompt should list character_response
    // headers AFTER character_principle/applied_insight/reasoning.
    for
      store <- VectorStoreAlgebra.inMemory
      _     <- store.ingest("d1", "some lore",
                            Map("character" -> "albus-dumbledore"))
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      llm   <- FakeLLMClient.of[IO](perspectiveResponse())
      _     <- PerspectiveShiftAgent.make[IO](store, llm, conv)
                 .ask("scenario", "Dumbledore", None)
      prompts <- llm.recordedPrompts
    yield
      val p = prompts.head.prompt
      val posPrinciple = p.lastIndexOf("[[ ## character_principle ## ]]")
      val posInsight   = p.lastIndexOf("[[ ## applied_insight ## ]]")
      val posReason    = p.lastIndexOf("[[ ## reasoning ## ]]")
      val posResponse  = p.lastIndexOf("[[ ## character_response ## ]]")
      assert(posPrinciple > 0 && posResponse > 0,
        "headers missing from prompt")
      assert(posResponse > posPrinciple,
        s"character_response must appear after character_principle. principle=$posPrinciple response=$posResponse")
      assert(posResponse > posInsight,
        s"character_response must appear after applied_insight")
      assert(posResponse > posReason,
        s"character_response must appear after reasoning")

  // -------------------------------------------------------------------------
  // AgentAlgebra router
  // -------------------------------------------------------------------------

  test("AgentAlgebra.ask routes by mode and returns UnknownMode for invalid"):
    for
      store <- VectorStoreAlgebra.inMemory
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      llm   <- FakeLLMClient.of[IO](fakeResponse())
      agent = AgentAlgebra.wire[IO](store, llm, conv)
      bad   <- agent.ask(
                 mode           = "nonexistent_mode",
                 primary        = "hi",
                 kwargs         = Map.empty,
                 conversationId = None,
               )
    yield assert(bad.isLeft && bad.left.exists(_.isInstanceOf[AgentError.UnknownMode]))

  test("AgentAlgebra.ask dispatches open_analysis correctly"):
    for
      store <- VectorStoreAlgebra.inMemory
      _     <- store.ingest("hp1", "lore", Map.empty)
      conv  <- ConversationStoreAlgebra.inMemory[IO]
      llm   <- FakeLLMClient.of[IO](fakeResponse(analysis = "ROUTED"))
      agent  = AgentAlgebra.wire[IO](store, llm, conv)
      out   <- agent.ask(
                 mode    = "open_analysis",
                 primary = "q",
                 kwargs  = Map.empty,
                 conversationId = None,
               )
    yield
      out match
        case Right(o) => assertEquals(o.fields.get("analysis"), Some("ROUTED"))
        case Left(e)  => fail(s"expected Right: $e")

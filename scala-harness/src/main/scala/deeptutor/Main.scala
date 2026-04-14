package deeptutor

import cats.effect.{ExitCode, IO, IOApp}
import com.comcast.ip4s.*
import org.http4s.ember.server.EmberServerBuilder
import org.http4s.server.middleware.{Logger as HttpLogger, CORS}
import org.typelevel.log4cats.slf4j.Slf4jLogger
import org.typelevel.log4cats.Logger

// =============================================================================
// Main — IOApp entry point (Cats Effect)
// =============================================================================
// IOApp is the Typelevel equivalent of `def main`.
// It wires up all resources (vector store, window factory, HTTP server) via
// the Resource bracket pattern, which guarantees clean shutdown on interrupt.

object Main extends IOApp:

  def run(args: List[String]): IO[ExitCode] =
    given logger: Logger[IO] = Slf4jLogger.getLogger[IO]

    for
      _      <- logger.info("deepTutor context harness starting...")

      // Build the vector store (in-memory; swap for chromadb() in production)
      store  <- VectorStoreAlgebra.inMemory

      // Seed with a few HP lore snippets so the server is queryable immediately
      _      <- seedLore(store)

      routes  = LoreRoutes(store).routes
      httpApp = HttpLogger.httpApp(logHeaders = true, logBody = false)(
                  CORS.policy.withAllowOriginAll(routes.orNotFound)
                )

      _ <- EmberServerBuilder
             .default[IO]
             .withHost(ipv4"0.0.0.0")
             .withPort(port"8080")
             .withHttpApp(httpApp)
             .build
             .use { server =>
               logger.info(s"Server started at ${server.address}") *>
               IO.never
             }
    yield ExitCode.Success

  // ---------------------------------------------------------------------------
  // Seed data — minimal HP lore for demo purposes
  // ---------------------------------------------------------------------------

  private def seedLore(store: VectorStoreAlgebra[IO]): IO[Unit] =
    val docs = List(
      ("hp-philosopher-stone",
       """Albus Dumbledore was the headmaster of Hogwarts School of Witchcraft and Wizardry.
         |He was widely considered the greatest wizard of his age, and was the only one
         |Voldemort ever feared. Dumbledore wielded the Elder Wand, the most powerful wand
         |ever made, and was the master of all three Deathly Hallows at one point.
         |
         |The Philosopher's Stone was a legendary alchemical substance said to be capable
         |of turning any metal into gold. It also produced the Elixir of Life, which made
         |the drinker immortal. Nicolas Flamel was its only known maker.""".stripMargin),

      ("hp-chamber-of-secrets",
       """The Chamber of Secrets was a hidden chamber deep beneath Hogwarts, built by
         |Salazar Slytherin. It housed a Basilisk — a giant serpent whose gaze is fatal
         |to those who look it directly in the eye. The entrance was concealed behind a
         |sink in Moaning Myrtle's bathroom on the second floor.
         |
         |Tom Riddle opened the Chamber in his fifth year, fifty years before Harry's second
         |year at Hogwarts. The Basilisk was controlled by the Heir of Slytherin.""".stripMargin),

      ("hp-deathly-hallows",
       """The Deathly Hallows are three magical objects created by Death (or, more plausibly,
         |by the three Peverell brothers): the Elder Wand, the Resurrection Stone, and the
         |Invisibility Cloak. Together they make the owner the Master of Death.
         |
         |Harry Potter became the true master of the Elder Wand without wielding it, by
         |disarming Draco Malfoy who had unknowingly won its allegiance from Dumbledore.
         |Harry chose to snap the Elder Wand after defeating Voldemort.""".stripMargin),
    )

    docs.traverse_ { case (docId, text) =>
      store.ingest(docId, text, Map("source" -> "HP canon")).void
    }

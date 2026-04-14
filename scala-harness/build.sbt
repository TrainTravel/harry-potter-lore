// deepTutor – Scala Typelevel Context Harness
// ============================================
// Stack: Cats Effect 3 · Http4s · Circe · fs2 · Refined

val scala3Version = "3.3.3"

lazy val root = project
  .in(file("."))
  .settings(
    name         := "deeptutor-harness",
    version      := "0.1.0-SNAPSHOT",
    scalaVersion := scala3Version,
    organization := "ai.deeptutor",

    // -----------------------------------------------------------------------
    // Dependencies
    // -----------------------------------------------------------------------
    libraryDependencies ++= Seq(
      // Cats Effect — purely functional async runtime
      "org.typelevel" %% "cats-effect"        % "3.5.4",
      "org.typelevel" %% "cats-core"          % "2.10.0",

      // fs2 — functional streaming (used for context chunk pipelines)
      "co.fs2"        %% "fs2-core"           % "3.10.2",
      "co.fs2"        %% "fs2-io"             % "3.10.2",

      // Http4s — purely functional HTTP server + client
      "org.http4s"    %% "http4s-ember-server" % "0.23.26",
      "org.http4s"    %% "http4s-ember-client" % "0.23.26",
      "org.http4s"    %% "http4s-circe"        % "0.23.26",
      "org.http4s"    %% "http4s-dsl"          % "0.23.26",

      // Circe — JSON codec derivation
      "io.circe"      %% "circe-core"          % "0.14.7",
      "io.circe"      %% "circe-generic"       % "0.14.7",
      "io.circe"      %% "circe-parser"        % "0.14.7",

      // Logging
      "org.typelevel" %% "log4cats-slf4j"      % "2.6.0",
      "ch.qos.logback" % "logback-classic"     % "1.5.6",

      // Test
      "org.typelevel" %% "munit-cats-effect-3" % "1.0.7" % Test,
      "org.scalameta" %% "munit"               % "0.7.29" % Test,
    ),

    // -----------------------------------------------------------------------
    // Compiler options — strict functional Scala
    // -----------------------------------------------------------------------
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Xfatal-warnings",
      "-language:higherKinds",
      "-language:implicitConversions",
    ),

    // -----------------------------------------------------------------------
    // Test framework
    // -----------------------------------------------------------------------
    testFrameworks += new TestFramework("munit.Framework"),
  )

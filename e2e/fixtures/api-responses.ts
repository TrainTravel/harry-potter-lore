/**
 * Canned AskResponse fixtures for route-intercepted E2E tests.
 *
 * Schema mirrors api/main.py AskResponse (lines 174-188).
 */

export interface Citation {
  doc_id: string;
  text: string;
  score: number;
}

export interface AskResponse {
  turn_id: string;
  answer: string;
  citations: Citation[];
  cost_usd: number;
  latency_ms: number;
  gaps: string;
  routed_mode: string | null;
  router_confidence: string | null;
}

const SAMPLE_CITATION: Citation = {
  doc_id: "hp_lore_chapter7",
  text: "The Deathly Hallows are three highly powerful magical objects supposedly created by Death.",
  score: 0.92,
};

// -- Mode-specific canned responses --

export const DEEP_RESEARCH: AskResponse = {
  turn_id: "test-deep-research-001",
  answer:
    "The Deathly Hallows consist of three legendary artifacts: the Elder Wand, the Resurrection Stone, and the Cloak of Invisibility. According to Beedle the Bard's tale, three brothers each received one gift from Death himself.",
  citations: [SAMPLE_CITATION],
  cost_usd: 0.001,
  latency_ms: 1200,
  gaps: "The exact mechanism by which Death created the Hallows is not explained in the text.",
  routed_mode: null,
  router_confidence: null,
};

export const GUIDED_LEARNING: AskResponse = {
  turn_id: "test-guided-learning-001",
  answer:
    "**Hint:** Think about what each Hallow represents — power, love, and humility.\n\n**Why it matters:** Understanding the symbolic meaning helps explain why Dumbledore was drawn to the Hallows in his youth.",
  citations: [SAMPLE_CITATION],
  cost_usd: 0.0008,
  latency_ms: 900,
  gaps: "",
  routed_mode: null,
  router_confidence: null,
};

export const EXAM_GRADER: AskResponse = {
  turn_id: "test-exam-grader-001",
  answer:
    "**Score:** 75/100 (PASS)\n\n**Critique:** Your answer correctly identifies the three Hallows but omits the Tale of the Three Brothers as their origin story.",
  citations: [SAMPLE_CITATION],
  cost_usd: 0.0007,
  latency_ms: 800,
  gaps: "",
  routed_mode: null,
  router_confidence: null,
};

export const OPEN_ANALYSIS: AskResponse = {
  turn_id: "test-open-analysis-001",
  answer:
    "Dumbledore's obsession with the Hallows reflects a deeper fear of death.\n\n**From the corpus:** Dumbledore sought the Resurrection Stone to see his sister Ariana again.\n\n**My analysis:** This parallels the second brother's fate in the tale — consumed by grief.",
  citations: [SAMPLE_CITATION],
  cost_usd: 0.001,
  latency_ms: 1100,
  gaps: "",
  routed_mode: null,
  router_confidence: null,
};

export const PERSPECTIVE_SHIFT: AskResponse = {
  turn_id: "test-perspective-shift-001",
  answer:
    "**Dumbledore's Principle:** It does not do to dwell on dreams and forget to live.\n\n**Applied to your situation:** A career change is not unlike finding the Mirror of Erised — do not stare so long at what might be that you forget to step through the door.\n\n**Why this maps:** Just as I eventually stopped seeking the Hallows and found purpose in teaching, sometimes wisdom means releasing the pursuit of power for something more meaningful.",
  citations: [SAMPLE_CITATION],
  cost_usd: 0.0009,
  latency_ms: 1000,
  gaps: "",
  routed_mode: null,
  router_confidence: null,
};

export const DEBATE: AskResponse = {
  turn_id: "test-debate-001",
  answer:
    "**For:** Snape's actions — protecting Harry, working as a double agent for years, and his dying wish to look into Lily's eyes — demonstrate a lifetime of redemption through love.\n\n**Against:** Snape bullied children for years, joined the Death Eaters voluntarily, and only switched sides because Voldemort targeted the woman he was obsessed with — not out of moral principle.\n\n**Verdict:** Snape is a tragic, complex figure. He was heroic in his actions but deeply flawed in his motivations and interpersonal conduct.",
  citations: [SAMPLE_CITATION],
  cost_usd: 0.0012,
  latency_ms: 1300,
  gaps: "",
  routed_mode: null,
  router_confidence: null,
};

export const SATIRICAL_PODCAST: AskResponse = {
  turn_id: "test-satirical-podcast-001",
  answer:
    'HOST: Welcome back to "Wizards in the Wild"! Today we\'re asking: could Quidditch survive a modern safety audit?\n\nGUEST: Absolutely not. You have 14-year-olds flying at 100mph on broomsticks with zero padding.\n\nHOST: And the Bludgers — those are basically assault with a deadly weapon.\n\n*The comedic tension lies in applying mundane bureaucratic logic to a magical sport where the rules were clearly written by someone who had never heard of OSHA.*',
  citations: [SAMPLE_CITATION],
  cost_usd: 0.0011,
  latency_ms: 1150,
  gaps: "",
  routed_mode: null,
  router_confidence: null,
};

// -- Auto-routed responses (mode="auto" with router metadata) --

export const AUTO_DEEP_RESEARCH: AskResponse = {
  ...DEEP_RESEARCH,
  turn_id: "test-auto-deep-001",
  routed_mode: "deep_research",
  router_confidence: "high",
};

export const AUTO_DEBATE: AskResponse = {
  ...DEBATE,
  turn_id: "test-auto-debate-001",
  routed_mode: "debate",
  router_confidence: "medium",
};

export const AUTO_PERSPECTIVE_SHIFT: AskResponse = {
  ...PERSPECTIVE_SHIFT,
  turn_id: "test-auto-perspective-001",
  routed_mode: "perspective_shift",
  router_confidence: "low",
};

export const AUTO_OFF_TOPIC: AskResponse = {
  turn_id: "test-auto-offtopic-001",
  answer:
    "That question is outside my area of expertise. I'm a Harry Potter lore agent — here's what I can help with:\n\n- Lore deep dives\n- Guided learning & quizzes\n- Character perspectives\n- Debates on HP topics\n- Satirical podcast scripts",
  citations: [],
  cost_usd: 0.0003,
  latency_ms: 400,
  gaps: "",
  routed_mode: "none",
  router_confidence: "high",
};

// Convenience: all auto-routed responses keyed by routed_mode
export const AUTO_RESPONSES: Record<string, AskResponse> = {
  deep_research: AUTO_DEEP_RESEARCH,
  guided_learning: { ...GUIDED_LEARNING, turn_id: "test-auto-guided-001", routed_mode: "guided_learning", router_confidence: "high" },
  exam_grader: { ...EXAM_GRADER, turn_id: "test-auto-exam-001", routed_mode: "exam_grader", router_confidence: "medium" },
  open_analysis: { ...OPEN_ANALYSIS, turn_id: "test-auto-analysis-001", routed_mode: "open_analysis", router_confidence: "high" },
  perspective_shift: AUTO_PERSPECTIVE_SHIFT,
  debate: AUTO_DEBATE,
  satirical_podcast: { ...SATIRICAL_PODCAST, turn_id: "test-auto-podcast-001", routed_mode: "satirical_podcast", router_confidence: "high" },
  none: AUTO_OFF_TOPIC,
};

/** Human-readable badge label for each routed_mode (from docs/lovable-intent-routing-ux.md:24-32) */
export const MODE_LABELS: Record<string, string> = {
  deep_research: "Lore Lookup",
  guided_learning: "Tutoring",
  exam_grader: "Grading",
  open_analysis: "Analysis",
  perspective_shift: "Character",
  debate: "Debate",
  satirical_podcast: "Comedy",
  none: "Off-topic",
};

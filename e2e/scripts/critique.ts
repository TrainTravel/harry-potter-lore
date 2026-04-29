/**
 * Critique runner — reads screenshots from the latest exploration run and
 * sends each screenshot + step context to Gemini 2.5 Pro for a design
 * critique. Output is a single markdown report at
 * ``<run-dir>/critique.md`` plus printed to stdout.
 *
 * Provider: Google Gemini via the generativelanguage.googleapis.com REST
 * endpoint. Reuses the existing GOOGLE_API_KEY (same one the lore agent
 * uses at runtime) so there's only one Google billing relationship to
 * manage.
 *
 * Usage::
 *
 *     export GOOGLE_API_KEY=AQ...
 *     cd e2e
 *     npx -y tsx scripts/critique.ts                            # latest run
 *     npx -y tsx scripts/critique.ts screenshots/2026-04-29...  # specific run
 *
 * Cost: ~$0.05-0.10 per run with Gemini 2.5 Pro (4 screenshots × ~3k
 * tokens input + ~2k tokens output critique). Roughly 1/4 the cost of
 * Claude Opus for similar quality.
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const E2E_ROOT = path.resolve(__dirname, "..");
const SCREENSHOTS_ROOT = path.join(E2E_ROOT, "screenshots");

const MODEL = process.env.CRITIQUE_MODEL ?? "gemini-2.5-pro";
const GEMINI_API_KEY =
  process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
  console.error("GOOGLE_API_KEY (or GEMINI_API_KEY) is required");
  process.exit(2);
}

// ---------------------------------------------------------------------------
// System prompt — mirrors .claude/skills/ui-design-critique.md but tightened
// for Gemini, which is hedgier than Claude on opinionated prompts. The
// "DO NOT" rules at the end act as guardrails against the most common drift
// patterns ("consider exploring", "could be improved", "best practices").
// ---------------------------------------------------------------------------
const SYSTEM_PROMPT = `You are a senior product designer with 15 years of experience shipping consumer products at the level of Notion, Linear, Stripe, and Anthropic. You critique interfaces directly. You don't soften, you don't preface with what's good, you don't suggest "consider also exploring."

You will receive a sequence of screenshots from a single user journey through an HP-lore RAG agent, plus the user task each screen represents. Critique the journey holistically.

Output structure (Markdown):

# UI Critique — <date>

## Three biggest issues (ranked by impact on user task completion)

For each issue:

ISSUE: <one-sentence description naming the design principle violated>
WHY IT MATTERS: <concrete user-task impact>
FIX: <one specific applicable change, not a direction>
EXAMPLE: <a product the reader has likely used that does this well>

## Per-screen notes

One short paragraph per screenshot, calling out anything specific to that screen that didn't fit the top three. Skip if a screen is fine.

## Ship first

ONE specific change to ship this week. ~50 words. The smallest edit with the biggest effect.

Constraints (non-negotiable):
- Three issues, not ten. Force prioritization.
- Distinguish AI-generated tells (default fonts, equal-weight everything, padding that's "consistent" but wrong, generic icons, stacked shadows+gradients+borders, centered content that shouldn't be centered, missing empty/error/loading states, buttons that don't feel pressable, forms with no inline validation, random typography scale) from genuine design bugs (information architecture, flow, density, accessibility).
- Don't suggest "redo the design system." Surgical, not architectural.
- Total output 400-700 words. Long enough to be useful, short enough to actually read.
- If a screen is genuinely fine, say so explicitly. Don't invent issues to fill space.

DO NOT use any of these phrases — they are signs of hedging:
- "Consider exploring..."   → say what to change instead
- "Could be improved"       → improved how, specifically
- "Looks good but..."       → don't lead with praise
- "It depends on the use case" → the user already gave you the use case
- "Best practices suggest..." → name the principle, not "best practices"
- "I notice the spacing is inconsistent" without naming the px value

Direct, opinionated, prioritized. If you find yourself writing one of those phrases, rewrite the sentence.`;

// ---------------------------------------------------------------------------
// Manifest types (mirrors explore.ts)
// ---------------------------------------------------------------------------

interface ManifestStep {
  key: string;
  task: string;
  notes?: string;
}

interface Manifest {
  version: number;
  url: string;
  capturedAt: string;
  steps: ManifestStep[];
}

async function findLatestRunDir(): Promise<string> {
  const entries = await fs.readdir(SCREENSHOTS_ROOT, { withFileTypes: true });
  const dirs = entries
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort()
    .reverse();
  if (dirs.length === 0) {
    throw new Error(`No runs in ${SCREENSHOTS_ROOT}. Run scripts/explore.ts first.`);
  }
  return path.join(SCREENSHOTS_ROOT, dirs[0]);
}

async function imageToBase64(p: string): Promise<string> {
  const buf = await fs.readFile(p);
  return buf.toString("base64");
}

// ---------------------------------------------------------------------------
// Gemini API call
// ---------------------------------------------------------------------------
//
// Note on max_output_tokens: Gemini 2.5 Pro reserves a chunk of the budget
// for "internal reasoning" before emitting text. With 2000 tokens, ~half is
// burned thinking and the actual response gets truncated. 6000 gives ~3000
// of usable text — plenty for the 400-700 word target.

interface GeminiPart {
  text?: string;
  inlineData?: { mimeType: string; data: string };
}

async function callGemini(
  systemPrompt: string,
  parts: GeminiPart[],
): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${GEMINI_API_KEY}`;

  const body = {
    systemInstruction: { parts: [{ text: systemPrompt }] },
    contents: [{ role: "user", parts }],
    generationConfig: {
      maxOutputTokens: 6000,
      temperature: 0.3,
    },
  };

  const resp = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Gemini API ${resp.status}: ${txt}`);
  }

  const json = (await resp.json()) as {
    candidates?: Array<{
      content?: { parts?: Array<{ text?: string }>; role?: string };
      finishReason?: string;
    }>;
  };

  const candidate = json.candidates?.[0];
  if (!candidate) {
    throw new Error(`Gemini returned no candidates: ${JSON.stringify(json)}`);
  }

  const text =
    candidate.content?.parts
      ?.filter((p) => typeof p.text === "string")
      ?.map((p) => p.text!)
      ?.join("\n") ?? "";

  if (!text) {
    throw new Error(
      `Gemini returned empty text. finishReason=${candidate.finishReason}. ` +
        `Likely the model burned all tokens on internal reasoning — bump maxOutputTokens.`,
    );
  }

  return text;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function run() {
  const runDir = process.argv[2]
    ? path.resolve(process.argv[2])
    : await findLatestRunDir();
  console.error(`Critiquing run: ${runDir}`);

  const manifest = JSON.parse(
    await fs.readFile(path.join(runDir, "manifest.json"), "utf-8"),
  ) as Manifest;

  const parts: GeminiPart[] = [];
  parts.push({
    text: `Live URL critiqued: ${manifest.url}\nCaptured at: ${manifest.capturedAt}\n\nThe following ${manifest.steps.length} screenshots are from a single user journey. For each, the user-task and notes describe what the user is trying to do at that moment. Critique the journey as one cohesive UX.\n`,
  });

  for (const step of manifest.steps) {
    const imgPath = path.join(runDir, `${step.key}.png`);
    const base64 = await imageToBase64(imgPath);
    parts.push({
      text: `\n---\n\nStep: ${step.key}\nUser task: ${step.task}\n${step.notes ? `Notes: ${step.notes}\n` : ""}`,
    });
    parts.push({
      inlineData: { mimeType: "image/png", data: base64 },
    });
  }

  const critique = await callGemini(SYSTEM_PROMPT, parts);

  const reportPath = path.join(runDir, "critique.md");
  const header = `<!-- Generated by e2e/scripts/critique.ts on ${new Date().toISOString()} -->\n<!-- Model: ${MODEL} | Run: ${runDir} -->\n\n`;
  await fs.writeFile(reportPath, header + critique);

  console.log(`Wrote critique → ${reportPath}`);
  console.log(`\n${critique}`);
}

run().catch((err) => {
  console.error("critique failed:", err);
  process.exit(1);
});

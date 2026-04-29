/**
 * Critique runner — reads screenshots from the latest exploration run and
 * sends each screenshot + step context to Claude Opus 4.7 for a design
 * critique. Output is a single markdown report at
 * ``<run-dir>/critique.md`` plus printed to stdout.
 *
 * Uses the ``ui-design-critique`` skill prompt (kept inline for portability;
 * if the skill at ``.claude/skills/ui-design-critique.md`` drifts, update
 * the inline copy below or read it from disk).
 *
 * Usage::
 *
 *     export ANTHROPIC_API_KEY=sk-ant-...
 *     cd e2e
 *     npx tsx scripts/critique.ts                            # latest run
 *     npx tsx scripts/critique.ts screenshots/2026-04-29...  # specific run
 *
 * Cost: ~$0.20-0.40 per run with Opus 4.7 (5 screenshots × ~3k tokens
 * input + ~2k tokens output critique).
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const E2E_ROOT = path.resolve(__dirname, "..");
const SCREENSHOTS_ROOT = path.join(E2E_ROOT, "screenshots");

const MODEL = process.env.CRITIQUE_MODEL ?? "claude-opus-4-5";
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

if (!ANTHROPIC_API_KEY) {
  console.error("ANTHROPIC_API_KEY is required");
  process.exit(2);
}

// The ui-design-critique skill prompt, inlined for portability. Mirror this
// against .claude/skills/ui-design-critique.md if either drifts.
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

Constraints:
- Three issues, not ten. Force prioritization.
- Distinguish AI-generated tells (default fonts, equal-weight everything, padding that's "consistent" but wrong, generic icons, stacked shadows+gradients+borders, centered content that shouldn't be centered, missing empty/error/loading states, buttons that don't feel pressable, forms with no inline validation, random typography scale) from genuine design bugs (IA, flow, density, accessibility).
- Don't suggest "redo the design system." Surgical, not architectural.
- Total output 400-700 words. Long enough to be useful, short enough to actually read.
- If the screen is genuinely fine, say so explicitly. Don't invent issues to fill space.

If your critique uses any of these phrases, rewrite: "consider exploring", "could be improved", "looks good but", "it depends on the use case", "best practices suggest", or "I notice the spacing is inconsistent" without naming the px value.`;

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

interface AnthropicContentBlock {
  type: "text" | "image";
  text?: string;
  source?: { type: "base64"; media_type: "image/png"; data: string };
}

async function callClaude(
  systemPrompt: string,
  userContent: AnthropicContentBlock[],
): Promise<string> {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": ANTHROPIC_API_KEY!,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 2000,
      system: systemPrompt,
      messages: [{ role: "user", content: userContent }],
    }),
  });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Anthropic API ${resp.status}: ${txt}`);
  }
  const json = (await resp.json()) as {
    content: Array<{ type: string; text?: string }>;
  };
  return json.content
    .filter((c) => c.type === "text")
    .map((c) => c.text ?? "")
    .join("\n");
}

async function run() {
  const runDir = process.argv[2]
    ? path.resolve(process.argv[2])
    : await findLatestRunDir();
  console.error(`Critiquing run: ${runDir}`);

  const manifest = JSON.parse(
    await fs.readFile(path.join(runDir, "manifest.json"), "utf-8"),
  ) as Manifest;

  // Build a single user message with all screenshots + their step context.
  const blocks: AnthropicContentBlock[] = [];
  blocks.push({
    type: "text",
    text: `Live URL critiqued: ${manifest.url}\nCaptured at: ${manifest.capturedAt}\n\nThe following ${manifest.steps.length} screenshots are from a single user journey. For each, the user-task and notes describe what the user is trying to do at that moment. Critique the journey as one cohesive UX.\n`,
  });

  for (const step of manifest.steps) {
    const imgPath = path.join(runDir, `${step.key}.png`);
    const base64 = await imageToBase64(imgPath);
    blocks.push({
      type: "text",
      text: `\n---\n\nStep: ${step.key}\nUser task: ${step.task}\n${step.notes ? `Notes: ${step.notes}\n` : ""}`,
    });
    blocks.push({
      type: "image",
      source: { type: "base64", media_type: "image/png", data: base64 },
    });
  }

  const critique = await callClaude(SYSTEM_PROMPT, blocks);

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

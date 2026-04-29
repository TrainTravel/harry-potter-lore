/**
 * UI exploration runner — drives the live Lovable UI through scripted
 * journeys, captures screenshots at key states, writes them to
 * ``e2e/screenshots/<timestamp>/`` along with a manifest describing each
 * step's user task.
 *
 * Mocks ``/ask`` with canned responses (like the existing test fixtures)
 * so screenshots are deterministic — the UI critique is about layout,
 * hierarchy, spacing, color, not about whether the LLM's answer happens
 * to read well today.
 *
 * Usage::
 *
 *     cd e2e
 *     npx tsx scripts/explore.ts
 *     # → screenshots saved under e2e/screenshots/<ISO timestamp>/
 *
 * Followed by ``scripts/critique.ts`` which sends those screenshots to
 * Claude for the actual critique.
 */

import { chromium, type Page } from "@playwright/test";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const E2E_ROOT = path.resolve(__dirname, "..");
const FRONTEND_URL =
  process.env.FRONTEND_URL ?? "https://lore-whisperer-ui.lovable.app";

// Canned mock responses for /ask. Diversified so each journey shows a
// representative response shape (deep_research, perspective_shift, etc.).
const MOCK_DEEP_RESEARCH = {
  turn_id: "explore-1",
  answer:
    "The Deathly Hallows are three legendary magical objects: the Elder Wand, the Resurrection Stone, and the Cloak of Invisibility. According to The Tale of the Three Brothers, Death himself created them and gave one to each of the Peverell brothers as a reward for tricking him at a river crossing.",
  citations: [
    {
      doc_id: "deathly-hallows",
      score: 0.92,
      text: "The Deathly Hallows are three legendary magical objects said to have been created by Death and given to the Peverell brothers...",
    },
    {
      doc_id: "albus-dumbledore",
      score: 0.71,
      text: "Dumbledore wielded the Elder Wand, the most powerful wand in existence...",
    },
  ],
  cost_usd: 0.0011,
  latency_ms: 3400,
  gaps: "",
  routed_mode: "deep_research",
  router_confidence: "high",
};

const MOCK_PERSPECTIVE_SHIFT = {
  turn_id: "explore-2",
  answer:
    "Ah, my dear friend, you ask me about a wound that does not heal cleanly. I once knew a young wizard with extraordinary gifts and a darkness that grew alongside them — Gellert Grindelwald — and I confused his brilliance for kinship. The lesson I carry is this: the strongest pull is not always the truest. Confidence in love is not bravado but the quiet wisdom to recognise who genuinely sees you.",
  citations: [
    {
      doc_id: "albus-dumbledore",
      score: 0.88,
      text: "Albus Percival Wulfric Brian Dumbledore was the headmaster of Hogwarts...",
    },
  ],
  cost_usd: 0.0316,
  latency_ms: 22000,
  gaps: "",
  routed_mode: "perspective_shift",
  router_confidence: "high",
};

interface JourneyStep {
  /** Unique key like "01-empty" — used as filename prefix */
  key: string;
  /** Human-readable user-task this step represents */
  task: string;
  /** Optional notes for the critique */
  notes?: string;
}

const MANIFEST_VERSION = 1;

interface Manifest {
  version: number;
  url: string;
  capturedAt: string;
  steps: JourneyStep[];
}

async function setupMocks(page: Page) {
  // Mock /ask — first call returns deep_research, subsequent calls cycle.
  let callIdx = 0;
  await page.route("**/ask", async (route) => {
    const body = callIdx === 0 ? MOCK_DEEP_RESEARCH : MOCK_PERSPECTIVE_SHIFT;
    callIdx += 1;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });

  // Hide Lovable's edit badge (overlays buttons in tests)
  await page.addInitScript(() => {
    const style = document.createElement("style");
    style.textContent = "#lovable-badge { display: none !important; }";
    (document.head || document.documentElement).appendChild(style);
  });
}

async function fullPageScreenshot(page: Page, outDir: string, key: string) {
  const file = path.join(outDir, `${key}.png`);
  await page.screenshot({ path: file, fullPage: true });
  return file;
}

async function run() {
  const timestamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-")
    .slice(0, 19);
  const outDir = path.join(E2E_ROOT, "screenshots", timestamp);
  await fs.mkdir(outDir, { recursive: true });

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();
  await setupMocks(page);

  const steps: JourneyStep[] = [];

  // ---- Journey 1: empty state ----
  await page.goto(`${FRONTEND_URL}/chat`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500); // let any animations settle
  await fullPageScreenshot(page, outDir, "01-empty-state");
  steps.push({
    key: "01-empty-state",
    task: "First-time visitor lands on /chat. They should immediately understand what the agent does and how to start.",
    notes: "Critique the visual hierarchy — what does the eye land on first? Is the input the focal point? Does the brand feel intentional or AI-generated?",
  });

  // ---- Journey 2: send a question via suggestion chip ----
  // Click any suggestion chip — first one available
  const firstChip = page
    .locator('[data-testid="suggestion-chip"], button')
    .filter({ hasText: /horcrux|hallow|wand|hogwarts|sort|dumbledore/i })
    .first();
  if (await firstChip.count()) {
    await firstChip.click();
    // Find the send button via input
    const input = page.getByPlaceholder(/ask|type|wand at the ready/i).first();
    if (await input.count()) {
      const sendBtn = input.locator(
        "xpath=following-sibling::button | ../button[last()]",
      );
      await sendBtn.first().click();
      // Wait for the assistant message
      await page
        .waitForSelector(
          '[data-testid="assistant-message"], [class*="message"]',
          { timeout: 10_000 },
        )
        .catch(() => null);
      await page.waitForTimeout(1500);
      await fullPageScreenshot(page, outDir, "02-mid-conversation");
      steps.push({
        key: "02-mid-conversation",
        task: "User has just sent a question and is reading the response. The answer should be the centerpiece; metadata, citations, and chrome should be supporting.",
        notes: "Critique whether the answer is easy to read. Are the cost/latency stats distracting? Does the citations panel feel attached or disconnected? Does the welcome hero get out of the way?",
      });
    }
  }

  // ---- Journey 3: open the mode override popover ----
  const overrideBtn = page
    .locator('button[aria-label*="Mode" i], button:has(svg.lucide-sliders-horizontal)')
    .first();
  if (await overrideBtn.count()) {
    await overrideBtn.click();
    await page.waitForTimeout(500);
    await fullPageScreenshot(page, outDir, "03-mode-popover");
    steps.push({
      key: "03-mode-popover",
      task: "Power user wants to override the auto-router and pick a specific mode. The popover should make options scannable.",
      notes: "Critique typography, spacing in the popover. Is the active option clearly indicated? Are descriptions helpful or noise?",
    });
    // Close popover
    await page.keyboard.press("Escape");
    await page.waitForTimeout(300);
  }

  // ---- Journey 4: speed toggle visible state ----
  // The Fast/Quality pills live near the input. Just screenshot the current state.
  // (The previous popover-close should have left the input area visible.)
  await page.waitForTimeout(300);
  await fullPageScreenshot(page, outDir, "04-input-area");
  steps.push({
    key: "04-input-area",
    task: "User looks at the input controls — speed toggle, mode override, send button. Each control's purpose should be evident.",
    notes: "Critique density and labeling of the input toolbar. Are the Fast/Quality pills self-explanatory?",
  });

  // ---- Write manifest ----
  const manifest: Manifest = {
    version: MANIFEST_VERSION,
    url: FRONTEND_URL,
    capturedAt: new Date().toISOString(),
    steps,
  };
  await fs.writeFile(
    path.join(outDir, "manifest.json"),
    JSON.stringify(manifest, null, 2),
  );

  // Print path so the next stage (critique.ts) can find the latest run
  console.log(outDir);

  await browser.close();
}

run().catch((err) => {
  console.error("explore failed:", err);
  process.exit(1);
});

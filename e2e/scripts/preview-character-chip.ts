/**
 * Slice-1 preview — captures the AFTER state for the chip-click bug fix.
 *
 * Drives the local hogwarts-chat dev server (default ``http://localhost:8080``),
 * mocks ``/ask`` so the response is deterministic, shuffles until the Sirius
 * character chip appears, clicks it, sends the question, and screenshots the
 * resulting assistant message.
 *
 * Confirms visually that the Sirius avatar (SB) — not Dumbledore (AD) — is
 * rendered, proving Slice 1 fixed the chip → character binding.
 *
 * Usage::
 *
 *     # Start the dev server first (in another terminal):
 *     cd ../hogwarts-chat && npm run dev
 *
 *     # Then capture the preview:
 *     cd e2e && npx tsx scripts/preview-character-chip.ts
 *
 *     # Override the URL if dev server runs on a different port:
 *     FRONTEND_URL=http://localhost:5173 npx tsx scripts/preview-character-chip.ts
 */

import { chromium, type Page } from "@playwright/test";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const E2E_ROOT = path.resolve(__dirname, "..");
const FRONTEND_URL = process.env.FRONTEND_URL ?? "http://localhost:8080";

const MOCK_SIRIUS_RESPONSE = {
  turn_id: "preview-sirius-1",
  answer:
    "Family, eh? Sometimes the ones who share your blood are the ones who " +
    "know you the least. Mine certainly did. The Blacks lived in a world of " +
    "dusty, dark notions about blood purity and power — I couldn't breathe " +
    "in it. I chose my own family, the ones who saw me: James, Remus, and " +
    "the Potters. They took me in. If your family doesn't accept you, find " +
    "the ones who do. Build your own Gryffindor.",
  citations: [
    {
      doc_id: "sirius-black",
      score: 0.91,
      text:
        "Sirius Orion Black was a pure-blood wizard, godfather of Harry " +
        "Potter, and a member of the Order of the Phoenix.",
    },
  ],
  cost_usd: 0.034,
  latency_ms: 18000,
  gaps: "",
  routed_mode: "perspective_shift",
  router_confidence: "high",
};

async function setupMocks(page: Page) {
  await page.route("**/ask", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_SIRIUS_RESPONSE),
    });
  });

  // Hide Lovable's edit badge so it doesn't overlay anything (no-op locally
  // but keeps parity with the deployed-Lovable explore.ts script).
  await page.addInitScript(() => {
    const style = document.createElement("style");
    style.textContent = "#lovable-badge { display: none !important; }";
    (document.head || document.documentElement).appendChild(style);
  });
}

async function shuffleUntilSirius(page: Page, maxAttempts = 12) {
  // The Character row picks one chip from the pool at random per shuffle.
  // The pool has 8 entries; expectation is finding Sirius within ~12 clicks.
  const sirius = page
    .locator('[data-testid="suggestion-chip"]')
    .filter({ hasText: /sirius/i });

  if (await sirius.count()) return sirius.first();

  const shuffleBtn = page.getByRole("button", {
    name: /show different prompt ideas|shuffle ideas/i,
  });

  for (let i = 0; i < maxAttempts; i++) {
    await shuffleBtn.click();
    await page.waitForTimeout(150);
    if (await sirius.count()) return sirius.first();
  }

  throw new Error(
    `Sirius chip never rotated in after ${maxAttempts} shuffles — ` +
      "either the pool is wrong or the chip text changed.",
  );
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
  const outDir = path.join(E2E_ROOT, "screenshots", `preview-${timestamp}`);
  await fs.mkdir(outDir, { recursive: true });

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();
  await setupMocks(page);

  await page.goto(`${FRONTEND_URL}/chat`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);

  // Empty-state baseline (shows whatever character chip rotated in)
  await fullPageScreenshot(page, outDir, "01-empty-state");

  // Find the Sirius chip (shuffle until it appears)
  const siriusChip = await shuffleUntilSirius(page);
  await fullPageScreenshot(page, outDir, "02-sirius-chip-visible");

  // Click the chip — fills the input and (post-fix) sets character state
  await siriusChip.click();
  await page.waitForTimeout(300);

  // Find the send button. The composer lives near the textarea; the send
  // button is the closest sibling button with the Send icon.
  const sendBtn = page
    .locator("button")
    .filter({ has: page.locator("svg.lucide-send") })
    .first();
  await sendBtn.click();

  // Wait for the assistant message
  await page.waitForSelector('[data-testid="assistant-message"]', {
    timeout: 10_000,
  });
  await page.waitForTimeout(1200);

  await fullPageScreenshot(page, outDir, "03-sirius-response");

  console.log(outDir);
  await browser.close();
  return outDir;
}

run()
  .then((outDir) => {
    console.log(`\nPreview screenshots saved to:\n  ${outDir}\n`);
  })
  .catch((err) => {
    console.error("preview failed:", err);
    process.exit(1);
  });

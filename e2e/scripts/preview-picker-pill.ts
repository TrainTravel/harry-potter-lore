/**
 * Slice-2 preview — captures the always-visible character picker pill.
 *
 * Drives the local hogwarts-chat dev server, mocks ``/ask``, and captures:
 *  1. Empty state with the pill visible (default character: Dumbledore)
 *  2. Pill popover open showing the full 9-character roster
 *  3. After clicking Hagrid in the popover — pill updates to "Speaking as Hagrid"
 *  4. Mid-conversation pill state (still visible, switchable)
 *
 * Usage::
 *
 *     # Start the dev server first:
 *     cd ../hogwarts-chat && npm run dev
 *
 *     # Then capture:
 *     cd e2e && npx tsx scripts/preview-picker-pill.ts
 */

import { chromium, type Page } from "@playwright/test";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const E2E_ROOT = path.resolve(__dirname, "..");
const FRONTEND_URL = process.env.FRONTEND_URL ?? "http://localhost:8080";

const MOCK_HAGRID_RESPONSE = {
  turn_id: "preview-hagrid-1",
  answer:
    "Aye, that's a tough spot you're in, but I'll tell yeh somethin' I " +
    "learned the hard way: family ain't always who you're born to. Some o' " +
    "the best folk in my life — Dumbledore, you lot, the creatures I look " +
    "after — none of 'em share my blood, an' they're more family than most. " +
    "Don't be afraid to build yer own.",
  citations: [
    {
      doc_id: "rubeus-hagrid",
      score: 0.88,
      text: "Rubeus Hagrid was the Keeper of Keys and Grounds at Hogwarts...",
    },
  ],
  cost_usd: 0.029,
  latency_ms: 16000,
  gaps: "",
  routed_mode: "perspective_shift",
  router_confidence: "high",
};

async function setupMocks(page: Page) {
  await page.route("**/ask", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_HAGRID_RESPONSE),
    });
  });
  await page.addInitScript(() => {
    const style = document.createElement("style");
    style.textContent = "#lovable-badge { display: none !important; }";
    (document.head || document.documentElement).appendChild(style);
  });
}

async function fullPageScreenshot(page: Page, outDir: string, key: string) {
  const file = path.join(outDir, `${key}.png`);
  await page.screenshot({ path: file, fullPage: true });
}

async function run() {
  const timestamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-")
    .slice(0, 19);
  const outDir = path.join(E2E_ROOT, "screenshots", `preview-pill-${timestamp}`);
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

  // 1. Empty state — pill should show "Speaking as Dumbledore" (current default)
  await fullPageScreenshot(page, outDir, "01-empty-with-pill");

  // 2. Open the pill popover
  const pill = page.getByTestId("character-picker-pill");
  await pill.click();
  await page.waitForTimeout(300);
  await fullPageScreenshot(page, outDir, "02-popover-open");

  // 3. Click Hagrid in the popover
  const hagridOption = page.getByTestId("character-option-rubeus-hagrid");
  await hagridOption.click();
  await page.waitForTimeout(300);
  // Click outside to close the popover
  await page.mouse.click(50, 50);
  await page.waitForTimeout(300);
  await fullPageScreenshot(page, outDir, "03-pill-shows-hagrid");

  // 4. Type a question and send to confirm character flows through
  const textarea = page.getByPlaceholder(/ask|type|wand at the ready/i).first();
  await textarea.fill("How do you handle family that doesn't accept you?");
  const sendBtn = page
    .locator("button")
    .filter({ has: page.locator("svg.lucide-send") })
    .first();
  await sendBtn.click();
  await page.waitForSelector('[data-testid="assistant-message"]', {
    timeout: 10_000,
  });
  await page.waitForTimeout(1000);
  await fullPageScreenshot(page, outDir, "04-mid-convo-pill-still-visible");

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

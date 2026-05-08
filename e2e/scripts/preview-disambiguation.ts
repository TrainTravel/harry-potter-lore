/**
 * Slice-3 preview — captures the perspective-shift disambiguation flow.
 *
 * Mocks ``/ask`` so the first call returns options (no character),
 * second call returns a perspective_shift answer (character bound).
 * Captures: free-form question → options chips → click → answer.
 *
 * Usage::
 *
 *     cd ../hogwarts-chat && npm run dev   # in another terminal
 *     cd e2e && npx tsx scripts/preview-disambiguation.ts
 */

import { chromium, type Page } from "@playwright/test";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const E2E_ROOT = path.resolve(__dirname, "..");
const FRONTEND_URL = process.env.FRONTEND_URL ?? "http://localhost:8080";

const MOCK_DISAMBIGUATION = {
  turn_id: "preview-disambig-1",
  answer: "Whose perspective would you like? Pick a character below.",
  citations: [],
  cost_usd: 0.0,
  latency_ms: 12,
  gaps: "",
  routed_mode: "perspective_shift",
  router_confidence: "high",
  options: [
    "albus-dumbledore",
    "hermione-granger",
    "harry-potter",
    "ron-weasley",
    "luna-lovegood",
    "minerva-mcgonagall",
    "severus-snape",
    "rubeus-hagrid",
    "sirius-black",
  ],
};

const MOCK_LUNA_RESPONSE = {
  turn_id: "preview-disambig-2",
  answer:
    "Imposter syndrome is a bit like a Wrackspurt — invisible, but it " +
    "settles in your brain and makes everything fuzzy. The trick is to " +
    "notice it without believing it. The thoughts you have about yourself " +
    "are not always the truest ones. I find it helps to write down the " +
    "evidence — actual proof you belong where you are. Wrackspurts hate " +
    "evidence.",
  citations: [
    {
      doc_id: "luna-lovegood",
      score: 0.86,
      text: "Luna Lovegood was known for her unusual perspective and serene confidence.",
    },
  ],
  cost_usd: 0.024,
  latency_ms: 14000,
  gaps: "",
  routed_mode: "perspective_shift",
  router_confidence: "high",
};

async function setupMocks(page: Page) {
  let callIdx = 0;
  await page.route("**/ask", async (route) => {
    const body = callIdx === 0 ? MOCK_DISAMBIGUATION : MOCK_LUNA_RESPONSE;
    callIdx += 1;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
  await page.addInitScript(() => {
    const style = document.createElement("style");
    style.textContent = "#lovable-badge { display: none !important; }";
    (document.head || document.documentElement).appendChild(style);
  });
}

async function shot(page: Page, outDir: string, key: string) {
  await page.screenshot({
    path: path.join(outDir, `${key}.png`),
    fullPage: true,
  });
}

async function run() {
  const timestamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-")
    .slice(0, 19);
  const outDir = path.join(
    E2E_ROOT,
    "screenshots",
    `preview-disambig-${timestamp}`,
  );
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

  // 1. Empty state — pill should NOT show a character (no default)
  await shot(page, outDir, "01-empty-no-default");

  // 2. Type a free-form perspective question
  const textarea = page.locator("textarea").first();
  await textarea.waitFor({ state: "visible", timeout: 10_000 });
  await textarea.fill("How do I deal with imposter syndrome?");
  const sendBtn = page
    .locator("button")
    .filter({ has: page.locator("svg.lucide-send") })
    .first();
  await sendBtn.click();

  // 3. Wait for the disambiguation response with options chips
  await page.waitForSelector('[data-testid^="option-character-"]', {
    timeout: 10_000,
  });
  await page.waitForTimeout(500);
  await shot(page, outDir, "02-options-chips");

  // 4. Click Luna in the options
  await page.getByTestId("option-character-luna-lovegood").click();
  await page.waitForTimeout(500);
  // 5. Wait for the Luna-voice response (replaces the disambiguation msg)
  await page.waitForFunction(
    () => {
      const messages = document.querySelectorAll(
        '[data-testid="assistant-message"]',
      );
      const last = messages[messages.length - 1];
      return last && /wrackspurt|invisible|fuzzy/i.test(last.textContent ?? "");
    },
    { timeout: 10_000 },
  );
  await page.waitForTimeout(800);
  await shot(page, outDir, "03-luna-response");

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

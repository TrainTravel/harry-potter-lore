import { test } from "../fixtures/test-base.js";

test("diagnose: dump landing-page DOM", async ({ page }) => {
  await page.goto("/chat");
  await page.waitForTimeout(2000);

  const bodyText = await page.locator("body").innerText();
  console.log("\n===== VISIBLE TEXT =====\n", bodyText, "\n===== END =====\n");

  const inputs = await page.locator("input, textarea, [contenteditable]").all();
  console.log(`\n===== ${inputs.length} input-like elements =====`);
  for (const el of inputs) {
    const tag = await el.evaluate((e) => e.tagName);
    const placeholder = await el.getAttribute("placeholder");
    const aria = await el.getAttribute("aria-label");
    const testId = await el.getAttribute("data-testid");
    console.log(`  <${tag}> placeholder="${placeholder}" aria-label="${aria}" testid="${testId}"`);
  }

  const buttons = await page.locator("button").all();
  console.log(`\n===== ${buttons.length} buttons (first 15) =====`);
  for (const el of buttons.slice(0, 15)) {
    const txt = (await el.textContent())?.trim().slice(0, 50);
    const aria = await el.getAttribute("aria-label");
    const testId = await el.getAttribute("data-testid");
    console.log(`  text="${txt}" aria-label="${aria}" testid="${testId}"`);
  }

  const testIds = await page.locator("[data-testid]").all();
  console.log(`\n===== ${testIds.length} elements with data-testid =====`);
  for (const el of testIds) {
    const testId = await el.getAttribute("data-testid");
    console.log(`  testid="${testId}"`);
  }
});

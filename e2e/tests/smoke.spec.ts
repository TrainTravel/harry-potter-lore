/**
 * Smoke tests — page loads, suggestion chips visible, basic send flow.
 *
 * Selectors based on live Lovable UI DOM (2026-04-26):
 *   - Suggestion chips: buttons under "Try one of these" paragraph
 *   - Input: textbox placeholder "Ask a question…"
 *   - Send: icon-only button sibling of the textbox
 */

import { test, expect } from "../fixtures/test-base.js";

test.describe("Smoke", () => {
  test("page loads and shows suggestion chips", async ({ page }) => {
    await page.goto("/chat");

    // "Try one of these" heading + at least one suggestion button
    await expect(
      page.getByText(/try one of these/i)
    ).toBeVisible({ timeout: 10_000 });

    // At least one suggestion chip button should be visible
    const chips = page.locator("button").filter({
      hasText: /snape|dumbledore|hermione|moral|hogwarts|horcrux|quidditch/i,
    });
    await expect(chips.first()).toBeVisible();
  });

  test("clicking a suggestion chip populates the input", async ({ page }) => {
    await page.goto("/chat");

    const chip = page.locator("button").filter({
      hasText: /snape|dumbledore|hermione|moral|hogwarts/i,
    }).first();
    await chip.waitFor({ timeout: 10_000 });

    const chipText = await chip.textContent();
    await chip.click();

    const input = page.getByPlaceholder(/ask a question/i);
    const inputValue = await input.inputValue();
    expect(inputValue.length).toBeGreaterThan(0);
  });

  test("submitting a question renders an assistant message", async ({
    page,
    sendMessage,
  }) => {
    await page.goto("/chat");
    await sendMessage("What are the Deathly Hallows?");

    // The canned response text should appear somewhere on the page
    await expect(
      page.getByText(/Deathly Hallows consist of three/i).first()
    ).toBeVisible({ timeout: 10_000 });
  });
});

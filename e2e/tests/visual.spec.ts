/**
 * Visual regression — screenshot baselines for empty state and response state.
 *
 * First run: generates baseline PNGs with `--update-snapshots`.
 * Subsequent runs: compare with 2% pixel tolerance.
 */

import { test, expect } from "../fixtures/test-base.js";

test.describe("Visual Regression", () => {
  test("empty chat state with suggestion chips", async ({ page }) => {
    await page.goto("/chat");
    await page.waitForLoadState("networkidle");

    // Wait for suggestion chips to render
    await expect(
      page.getByText(/try one of these/i)
    ).toBeVisible({ timeout: 10_000 });

    await expect(page).toHaveScreenshot("empty-chat-state.png", {
      maxDiffPixelRatio: 0.02,
      fullPage: true,
    });
  });

  test("chat with one response", async ({ page, sendMessage }) => {
    await page.goto("/chat");
    await sendMessage("What are the Deathly Hallows?");

    // Let animations settle
    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot("chat-with-response.png", {
      maxDiffPixelRatio: 0.02,
      fullPage: true,
    });
  });
});

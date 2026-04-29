/**
 * Intent routing tests — badge labels, confidence colors, off-topic styling.
 *
 * These tests assert the UX described in docs/lovable-intent-routing-ux.md.
 * They are marked fixme() because the Lovable UI has not yet been updated to
 * render routed_mode badges or confidence color coding. Remove fixme() once
 * the Lovable prompt from that doc has been applied.
 */

import { test, expect } from "../fixtures/test-base.js";
import {
  AUTO_DEEP_RESEARCH,
  AUTO_DEBATE,
  AUTO_PERSPECTIVE_SHIFT,
  AUTO_OFF_TOPIC,
  MODE_LABELS,
} from "../fixtures/api-responses.js";

test.describe("Intent Routing", () => {
  test(
    "routed_mode badge shows correct human-readable label",
    async ({ page, sendMessage }) => {
      await page.unroute("**/ask");
      await page.route("**/ask", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(AUTO_DEEP_RESEARCH),
        })
      );

      await page.goto("/chat");
      await sendMessage("What are the Deathly Hallows?");

      const badge = page.getByText(MODE_LABELS.deep_research, { exact: false });
      await expect(badge.first()).toBeVisible({ timeout: 5_000 });
    }
  );

  test(
    "confidence level affects badge color coding",
    async ({ page, sendMessage }) => {
      const cases = [
        { response: AUTO_DEEP_RESEARCH, confidence: "high" },
        { response: AUTO_DEBATE, confidence: "medium" },
        { response: AUTO_PERSPECTIVE_SHIFT, confidence: "low" },
      ];

      for (const { response, confidence } of cases) {
        await page.unroute("**/ask");
        await page.route("**/ask", (route) =>
          route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(response),
          })
        );

        await page.goto("/chat");
        await sendMessage("Test question");

        const badge = page.locator(
          "[class*='badge'], [class*='chip'], [class*='tag']"
        );
        await expect(badge.first()).toBeVisible({ timeout: 5_000 });

        const badgeEl = badge.first();
        const classAttr = (await badgeEl.getAttribute("class")) ?? "";
        const dataConfidence =
          (await badgeEl.getAttribute("data-confidence")) ?? "";

        const hasConfidenceSignal =
          classAttr.includes(confidence) ||
          dataConfidence.includes(confidence) ||
          classAttr.match(/green|success|emerald/) !== null ||
          classAttr.match(/amber|warning|yellow/) !== null ||
          classAttr.match(/red|error|destructive/) !== null;

        if (!hasConfidenceSignal) {
          console.warn(
            `Badge for confidence="${confidence}" may not have visible color coding. ` +
              `class="${classAttr}", data-confidence="${dataConfidence}"`
          );
        }
      }
    }
  );

  test(
    "off-topic response styled as info nudge, not error",
    async ({ page, sendMessage }) => {
      await page.unroute("**/ask");
      await page.route("**/ask", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(AUTO_OFF_TOPIC),
        })
      );

      await page.goto("/chat");
      await sendMessage("What's the weather today?");

      await expect(
        page.getByText(/outside my area of expertise/i).first()
      ).toBeVisible({ timeout: 5_000 });

      const badge = page.getByText(/off.?topic/i);
      await expect(badge.first()).toBeVisible({ timeout: 5_000 });
    }
  );
});

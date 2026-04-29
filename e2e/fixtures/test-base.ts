/**
 * Shared Playwright fixture: auto-intercepts POST /ask with canned responses
 * and provides a `sendMessage` helper.
 *
 * Selectors are based on the live Lovable UI DOM snapshot (2026-04-26):
 *   - Input: textbox with placeholder "Ask a question…"
 *   - Send button: icon-only button adjacent to the textbox (no text/aria-label)
 *   - Suggestion chips: buttons with question text under "Try one of these"
 *   - Assistant messages: identified by DOM structure after send
 */

import { test as base, expect } from "@playwright/test";
import { DEEP_RESEARCH, type AskResponse } from "./api-responses.js";

/** Options the fixture accepts */
export interface LoreFixtureOptions {
  /** Canned response to return for the next /ask call */
  cannedResponse: AskResponse;
}

/**
 * Extended test fixture that intercepts /ask and provides helpers.
 *
 * Usage in spec files:
 *   import { test, expect } from "../fixtures/test-base.js";
 *   test("my test", async ({ page, sendMessage }) => { ... });
 */
export const test = base.extend<
  LoreFixtureOptions & { sendMessage: (text: string) => Promise<void> }
>({
  cannedResponse: [DEEP_RESEARCH, { option: true }],

  page: async ({ page, cannedResponse }, use) => {
    // Intercept any request to /ask with the canned response
    await page.route("**/ask", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(cannedResponse),
      });
    });

    // Lovable-deployed previews inject a fixed-position "Edit with Lovable"
    // badge that intercepts pointer events on adjacent buttons. Hide it on
    // every navigation so it can't block clicks during tests.
    await page.addInitScript(() => {
      const style = document.createElement("style");
      style.textContent = "#lovable-badge { display: none !important; }";
      (document.head || document.documentElement).appendChild(style);
    });

    await use(page);
  },

  sendMessage: async ({ page }, use) => {
    const fn = async (text: string) => {
      // Input: textbox in the chat composer. Match either the original
      // "Ask a question…" placeholder or the regen "Ask anything about the
      // Wizarding World…" — both are valid Lovable wordings.
      const input = page.getByPlaceholder(/ask (anything|a question)/i);
      await input.fill(text);

      // Send button: icon-only button adjacent to the textbox in DOM order.
      const sendBtn = input.locator(
        "xpath=following-sibling::button | ../button[last()]"
      );

      // Remove the Lovable preview badge if present — it overlays the send
      // button and blocks pointer events on Lovable-deployed pages.
      await page.evaluate(() => {
        document
          .querySelectorAll('#lovable-badge, [id^="lovable-badge"]')
          .forEach((el) => el.remove());
      });

      await sendBtn.first().click();

      // Wait for a response to appear. The Lovable UI renders assistant
      // messages in the chat area. After route interception fulfills /ask,
      // the UI should render the canned answer text.
      await page.waitForSelector(
        '[class*="message"], [class*="response"], [class*="chat"] p, [class*="prose"]',
        { timeout: 10_000 },
      );
    };
    await use(fn);
  },
});

export { expect } from "@playwright/test";

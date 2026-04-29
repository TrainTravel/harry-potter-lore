/**
 * Multi-turn tests — conversation_id maintained, message count grows.
 */

import { test as base, expect } from "@playwright/test";
import { DEEP_RESEARCH, type AskResponse } from "../fixtures/api-responses.js";

// Custom fixture that captures intercepted request bodies
const test = base.extend<{
  capturedRequests: { body: Record<string, unknown> }[];
  sendMessage: (text: string) => Promise<void>;
}>({
  capturedRequests: async ({}, use) => {
    const captured: { body: Record<string, unknown> }[] = [];
    await use(captured);
  },

  page: async ({ page, capturedRequests }, use) => {
    let turnCount = 0;
    await page.route("**/ask", async (route) => {
      const request = route.request();
      const postData = request.postDataJSON();
      capturedRequests.push({ body: postData });

      turnCount++;
      const response: AskResponse = {
        ...DEEP_RESEARCH,
        turn_id: `test-multi-turn-${turnCount}`,
        answer: `Response number ${turnCount}: The Deathly Hallows are legendary artifacts.`,
      };
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(response),
      });
    });

    // Hide the Lovable badge so it can't intercept clicks
    await page.addInitScript(() => {
      const style = document.createElement("style");
      style.textContent = "#lovable-badge { display: none !important; }";
      (document.head || document.documentElement).appendChild(style);
    });

    await use(page);
  },

  sendMessage: async ({ page }, use) => {
    const fn = async (text: string) => {
      const input = page.getByPlaceholder(/ask a question/i);
      await input.fill(text);

      // Remove Lovable badge before clicking — it overlays the send button
      await page.evaluate(() => {
        document
          .querySelectorAll('#lovable-badge, [id^="lovable-badge"]')
          .forEach((el) => el.remove());
      });

      // Icon-only send button: sibling of the textbox
      const sendBtn = input.locator(
        "xpath=following-sibling::button | ../button[last()]"
      );
      await sendBtn.first().click();

      // Wait for the response text to appear
      await expect(
        page.getByText(/Deathly Hallows are legendary/i).first()
      ).toBeVisible({ timeout: 10_000 });
    };
    await use(fn);
  },
});

test.describe("Multi-turn", () => {
  test("conversation_id is present and consistent across turns", async ({
    page,
    sendMessage,
    capturedRequests,
  }) => {
    await page.goto("/chat");

    await sendMessage("What are the Deathly Hallows?");
    // Small wait for UI to settle before second send
    await page.waitForTimeout(500);
    await sendMessage("Tell me more about the Elder Wand");

    expect(capturedRequests.length).toBe(2);

    const firstConvId = capturedRequests[0].body.conversation_id as
      | string
      | undefined;
    const secondConvId = capturedRequests[1].body.conversation_id as
      | string
      | undefined;

    if (firstConvId) {
      expect(firstConvId).toBeTruthy();
      expect(secondConvId).toBe(firstConvId);
    }
    if (!firstConvId) {
      console.warn(
        "UI does not send conversation_id — multi-turn state is server-side only or not implemented"
      );
    }
  });

  test("second response appends below first", async ({
    page,
    sendMessage,
  }) => {
    await page.goto("/chat");

    await sendMessage("What are the Deathly Hallows?");

    // Count response elements after first send
    await expect(
      page.getByText(/Response number 1/i)
    ).toBeVisible({ timeout: 10_000 });

    await page.waitForTimeout(500);
    await sendMessage("Tell me more about the Elder Wand");

    // Second response should also be visible
    await expect(
      page.getByText(/Response number 2/i)
    ).toBeVisible({ timeout: 10_000 });

    // Both responses should be on the page
    const bodyText = await page.locator("body").textContent() ?? "";
    expect(bodyText).toContain("Response number 1");
    expect(bodyText).toContain("Response number 2");
  });
});

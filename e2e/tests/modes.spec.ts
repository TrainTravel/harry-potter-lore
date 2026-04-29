/**
 * Mode rendering tests — each mode's distinctive output structure renders correctly.
 */

import { test, expect } from "../fixtures/test-base.js";
import {
  DEEP_RESEARCH,
  GUIDED_LEARNING,
  EXAM_GRADER,
  OPEN_ANALYSIS,
  PERSPECTIVE_SHIFT,
  DEBATE,
  SATIRICAL_PODCAST,
  AUTO_DEEP_RESEARCH,
  type AskResponse,
} from "../fixtures/api-responses.js";

interface ModeCase {
  name: string;
  response: AskResponse;
  /** Strings that should appear in the rendered output */
  expectedContent: RegExp[];
}

const MODE_CASES: ModeCase[] = [
  {
    name: "deep_research",
    response: DEEP_RESEARCH,
    expectedContent: [/Deathly Hallows/, /Elder Wand|Resurrection Stone|Cloak/],
  },
  {
    name: "guided_learning",
    response: GUIDED_LEARNING,
    expectedContent: [/Hint/, /Why it matters/],
  },
  {
    name: "exam_grader",
    response: EXAM_GRADER,
    expectedContent: [/Score.*75\/100/, /PASS/, /Critique/],
  },
  {
    name: "open_analysis",
    response: OPEN_ANALYSIS,
    expectedContent: [/From the corpus/, /My analysis/],
  },
  {
    name: "perspective_shift",
    response: PERSPECTIVE_SHIFT,
    expectedContent: [/Principle/, /Applied/],
  },
  {
    name: "debate",
    response: DEBATE,
    expectedContent: [/For:/, /Against:/, /Verdict:/],
  },
  {
    name: "satirical_podcast",
    response: SATIRICAL_PODCAST,
    expectedContent: [/HOST:|GUEST:/, /Quidditch|broomstick/i],
  },
  {
    name: "auto (routed to deep_research)",
    response: AUTO_DEEP_RESEARCH,
    expectedContent: [/Deathly Hallows/],
  },
];

for (const { name, response, expectedContent } of MODE_CASES) {
  test(`mode: ${name} — renders distinctive structure`, async ({
    page,
    sendMessage,
  }) => {
    await page.unroute("**/ask");
    await page.route("**/ask", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(response),
      })
    );

    await page.goto("/chat");
    await sendMessage("Test question for mode");

    // Wait for the answer text to appear on the page
    const firstPattern = expectedContent[0];
    await expect(
      page.getByText(firstPattern).first()
    ).toBeVisible({ timeout: 10_000 });

    // Verify all expected patterns are present in the page
    const bodyText = await page.locator("body").textContent() ?? "";
    for (const pattern of expectedContent) {
      expect(bodyText).toMatch(pattern);
    }
  });
}

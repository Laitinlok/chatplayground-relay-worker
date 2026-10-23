import { describe, expect, it } from "vitest";
import {
  needsCurrentWebSearch,
  sanitizeSearchQuery,
} from "../src/utils/search-query";

describe("sanitizeSearchQuery", () => {
  it("removes Agora wrapper tags and sent-date metadata", () => {
    expect(
      sanitizeSearchQuery(
        "</agora_user_message sent_date=2026-04-10T10:30:00Z>Latest climate report</agiora_user_message>",
      ),
    ).toBe("Latest climate report");
  });

  it.each([
    "latest viral TikTok food hacks",
    "recent stock price for NVDA",
    "live weather in Tokyo",
    "verify whether the release is official",
    "find current documentation on the web",
  ])("recognizes information that requires web search: %s", (prompt) => {
    expect(needsCurrentWebSearch(prompt)).toBe(true);
  });

  it("does not require search for stable knowledge", () => {
    expect(needsCurrentWebSearch("Explain how binary search works.")).toBe(
      false,
    );
  });
});

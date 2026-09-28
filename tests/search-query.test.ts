import { describe, expect, it } from "vitest";
import { sanitizeSearchQuery } from "../src/utils/search-query";

describe("sanitizeSearchQuery", () => {
  it("removes Agora wrapper tags and sent-date metadata", () => {
    expect(
      sanitizeSearchQuery(
        "</agora_user_message sent_date=2026-04-10T10:30:00Z>Latest climate report</agiora_user_message>",
      ),
    ).toBe("Latest climate report");
  });

  it("preserves ordinary query text and trims to the search limit", () => {
    expect(sanitizeSearchQuery("  current   weather  ")).toBe("current weather");
    expect(sanitizeSearchQuery("x".repeat(400))).toHaveLength(300);
  });
});

import { afterEach, describe, expect, it, vi } from "vitest";
import { webSearch } from "../src/utils/web-search";

afterEach(() => vi.restoreAllMocks());

describe("Cloudflare Search web search", () => {
  it("normalizes aggregated results and sends the configured token", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({
        results: [
          {
            title: "Climate report",
            url: "https://example.org/climate",
            description: "A current climate report.",
            engine: "brave",
          },
        ],
      }),
    );

    await expect(
      webSearch("latest climate report", {
        url: "https://search.example.test",
        token: "test-search-token",
        count: 7,
      }),
    ).resolves.toEqual([
      {
        title: "Climate report",
        url: "https://example.org/climate",
        snippet: "A current climate report.",
      },
    ]);

    const [requestUrl, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
    expect(requestUrl).toBe("https://search.example.test/search");
    expect(init.method).toBe("POST");
    expect(new Headers(init.headers).get("content-type")).toContain(
      "application/x-www-form-urlencoded",
    );
    expect(String(init.body)).toContain("q=latest+climate+report");
    expect(String(init.body)).toContain("token=test-search-token");
  });

  it("normalizes Cloudflare Search content fields and safely caps result counts", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({
        results: [
          ...Array.from({ length: 6 }, (_, index) => ({
            title: `Result ${index + 1}`,
            url: `https://example.org/${index + 1}`,
            content: `Description ${index + 1}`,
          })),
          { title: "Unsafe", url: "javascript:alert(1)" },
        ],
      }),
    );

    const results = await webSearch("query", {
      url: "https://search.example.test",
      count: 5,
    });
    expect(results).toHaveLength(5);
    expect(results[0]).toEqual({
      title: "Result 1",
      url: "https://example.org/1",
      snippet: "Description 1",
    });
    expect(results.some((result) => result.title === "Unsafe")).toBe(false);
  });

  it("rejects an invalid search URL and returns no results", async () => {
    const fetch = vi.spyOn(globalThis, "fetch");
    await expect(webSearch("query", { url: "file:///tmp/search" })).resolves.toEqual([]);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("returns no results when the search service fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("unauthorized", { status: 401 }),
    );

    await expect(
      webSearch("query", { url: "https://search.example.test" }),
    ).resolves.toEqual([]);
  });
});

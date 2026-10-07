import { afterEach, describe, expect, it, vi } from "vitest";
import { webSearch } from "../src/utils/web-search";

afterEach(() => vi.restoreAllMocks());

describe("Cloudflare Search web search", () => {
  it("normalizes aggregated results and sends the configured Bearer token", async () => {
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
    expect(new Headers(init.headers).get("authorization")).toBe(
      "Bearer test-search-token",
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

  it("falls back to Jina Markdown DuckDuckGo results when Cloudflare fails", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("search.example.test")) {
        return new Response("unauthorized", { status: 401 });
      }
      return new Response(
        [
          "Title: query at DuckDuckGo",
          "",
          "Markdown Content:",
          "## [OpenAI News](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fopenai.com%2Fnews%2F&rut=abc)",
          "",
          "[Stay up to speed on AI.](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fopenai.com%2Fnews%2F&rut=abc)",
          "",
          "## [Reuters OpenAI](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.reuters.com%2Ftechnology%2Fopenai%2F&rut=def)",
          "",
          "Latest OpenAI stories from Reuters.",
        ].join("\n"),
      );
    });

    const results = await webSearch("query", { url: "https://search.example.test" });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toEqual({
      title: "OpenAI News",
      url: "https://openai.com/news/",
      snippet: "Stay up to speed on AI.",
    });
    expect(results[1]?.url).toBe("https://www.reuters.com/technology/openai/");
    expect(fetchMock.mock.calls.map(([url]) => String(url)).some((url) => url.includes("r.jina.ai"))).toBe(true);
  });

  it("rejects an invalid search URL and falls back to Jina/DuckDuckGo", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("", { status: 500 }),
    );
    await expect(webSearch("query", { url: "not-a-url" })).resolves.toEqual([]);
    expect(fetchMock).toHaveBeenCalled();
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("r.jina.ai");
  });

  it("tries GET when Cloudflare POST fails with 500", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("search.example.test") && (init?.method ?? "GET") === "POST") {
        return new Response("worker error", { status: 500 });
      }
      if (url.includes("search.example.test")) {
        return Response.json({
          results: [
            {
              title: "Via GET",
              url: "https://example.org/get",
              description: " recovered",
            },
          ],
        });
      }
      return new Response("", { status: 500 });
    });

    const results = await webSearch("query", {
      url: "https://search.example.test",
      token: "tok",
    });
    expect(results).toEqual([
      { title: "Via GET", url: "https://example.org/get", snippet: "recovered" },
    ]);
    expect(fetchMock.mock.calls.some(([, init]) => (init?.method ?? "GET") === "GET")).toBe(true);
  });
});

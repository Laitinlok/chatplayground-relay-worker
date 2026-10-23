import { afterEach, describe, expect, it, vi } from "vitest";
import { webSearch } from "../src/utils/web-search";

afterEach(() => vi.restoreAllMocks());

describe("Cloudflare Search web search", () => {
  it("accepts configured provider results taking longer than 500 ms", async () => {
    vi.useFakeTimers();
    try {
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        if (String(input).startsWith("https://search.example.test/")) {
          await new Promise((resolve) => setTimeout(resolve, 750));
          return Response.json({
            results: [
              { title: "Slow result", url: "https://example.org/result" },
            ],
          });
        }
        return new Response("Unavailable", { status: 503 });
      });
      const result = webSearch("latest news", {
        url: "https://search.example.test",
      });
      await vi.advanceTimersByTimeAsync(750);
      expect(await result).toEqual([
        {
          title: "Slow result",
          url: "https://example.org/result",
          snippet: "",
        },
      ]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("cancels losing providers after a useful response", async () => {
    const signals: AbortSignal[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      if (init?.signal) signals.push(init.signal);
      if (String(input) === "https://search.example.test/search") {
        return Response.json({
          results: [{ title: "Result", url: "https://example.org/result" }],
        });
      }
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener(
          "abort",
          () => reject(new Error("aborted")),
          { once: true },
        );
      });
    });
    const result = await webSearch("query", {
      url: "https://search.example.test",
    });
    expect(result).toHaveLength(1);
    expect(signals).toHaveLength(4);
    expect(signals.every((signal) => signal.aborted)).toBe(true);
  });

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

    const [requestUrl, init] = vi.mocked(fetch).mock.calls[0] as [
      string,
      RequestInit,
    ];
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

  it("parses the Cloudflare Web Search API items envelope", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({
        items: [
          {
            title: "Workers Web Search",
            url: "https://developers.cloudflare.com/web-search/",
            description: "Cloudflare's Web Search API returns items.",
          },
          {
            title: "Alternate URL field",
            link: "https://example.org/alternate",
            description: "Link-shaped result.",
          },
        ],
        metadata: { query: "Cloudflare web search" },
      }),
    );

    await expect(
      webSearch("Cloudflare web search", {
        url: "https://search.example.test",
        provider: "configured",
      }),
    ).resolves.toEqual([
      {
        title: "Workers Web Search",
        url: "https://developers.cloudflare.com/web-search/",
        snippet: "Cloudflare's Web Search API returns items.",
      },
      {
        title: "Alternate URL field",
        url: "https://example.org/alternate",
        snippet: "Link-shaped result.",
      },
    ]);
    expect(vi.mocked(fetch).mock.calls).toHaveLength(2);
    expect(
      vi.mocked(fetch).mock.calls.every(([input]) =>
        String(input).startsWith("https://search.example.test/"),
      ),
    ).toBe(true);
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
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (input) => {
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

    const results = await webSearch("query", {
      url: "https://search.example.test",
    });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toEqual({
      title: "OpenAI News",
      url: "https://openai.com/news/",
      snippet: "Stay up to speed on AI.",
    });
    expect(results[1]?.url).toBe("https://www.reuters.com/technology/openai/");
    expect(
      fetchMock.mock.calls
        .map(([url]) => String(url))
        .some((url) => url.includes("r.jina.ai")),
    ).toBe(true);
  });

  it("rejects an invalid search URL and falls back to Jina/DuckDuckGo", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("", { status: 500 }));
    await expect(webSearch("query", { url: "not-a-url" })).resolves.toEqual([]);
    expect(fetchMock).toHaveBeenCalled();
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("r.jina.ai");
  });

  it("tries GET when Cloudflare POST fails with 500", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (input, init) => {
        const url = String(input);
        if (
          url.includes("search.example.test") &&
          (init?.method ?? "GET") === "POST"
        ) {
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
      {
        title: "Via GET",
        url: "https://example.org/get",
        snippet: "recovered",
      },
    ]);
    expect(
      fetchMock.mock.calls.some(
        ([, init]) => (init?.method ?? "GET") === "GET",
      ),
    ).toBe(true);
  });
});

import { describe, expect, it } from "vitest";
import {
  isSearchTool,
  isFetchTool,
  prioritizeSearchTools,
  createSearchToolCall,
  rankToolsByEmbedding,
} from "../src/utils/tool-selector";
import type { OpenAITool } from "../src/utils/tool-shim";

function tool(
  name: string,
  properties: Record<string, unknown>,
  description?: string,
): OpenAITool {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters: { type: "object", properties },
    },
  };
}

describe("embedding-assisted tool ranking", () => {
  it("ranks relevant tools and reports a confident match", async () => {
    const relevant = tool("create_calendar_event", {
      title: { type: "string" },
    });
    const unrelated = tool("search_memories", {
      query: { type: "string" },
    }, "Search personal conversation memories.");
    const ai = {
      run: async (_model: string, input: { text: string[] }) => {
        expect(input.text).toHaveLength(3);
        return { data: [[1, 0], [0.1, 0.99], [0.98, 0.1]] };
      },
    };

    const ranked = await rankToolsByEmbedding(
      "Schedule a meeting for tomorrow",
      [unrelated, relevant],
      ai,
    );
    expect(ranked?.tools.map((item) => item.function.name)).toEqual([
      "create_calendar_event",
      "search_memories",
    ]);
    expect(ranked?.confident).toBe(true);
  });

  it("leaves ambiguous rankings and embedding failures to normal routing", async () => {
    const tools = [
      tool("create_event", { title: { type: "string" } }),
      tool("update_event", { id: { type: "string" } }),
    ];
    const ambiguous = await rankToolsByEmbedding(
      "Change the event",
      tools,
      { run: async () => ({ data: [[1, 0], [0.9, 0.1], [0.9, 0.1]] }) },
    );
    expect(ambiguous?.confident).toBe(false);
    await expect(
      rankToolsByEmbedding("Change the event", tools),
    ).resolves.toBeNull();

    const weak = await rankToolsByEmbedding("Change the event", tools, {
      run: async () => ({ data: [[1, 0], [0.2, 0.98], [0.1, 0.99]] }),
    });
    expect(weak?.confident).toBe(false);

    await expect(
      rankToolsByEmbedding("Change the event", tools, {
        run: async () => {
          throw new Error("AI unavailable");
        },
      }),
    ).resolves.toBeNull();
  });
});

describe("web tool classification", () => {
  it.each([
    ["web-browsing", "query"],
    ["search_news", "search_query"],
    ["search_web_news", "search_query"],
    ["BROWSING_internet", "query"],
    ["WEB_LOOKUP", "query"],
  ])("recognizes %s with %s", (name, parameter) => {
    expect(isSearchTool(tool(name, { [parameter]: { type: "string" } }))).toBe(
      true,
    );
  });

  it.each([
    ["database", { query: { type: "string" } }],
    ["web_lookup", { url: { type: "string" } }],
    ["search_web_news", {}],
    ["browser", { query: { type: "string" } }],
  ])(
    "does not classify %s without both name and parameter matches",
    (name, properties) => {
      expect(isSearchTool(tool(name, properties))).toBe(false);
    },
  );

  it("does not mistake conversation search tools for web search", () => {
    expect(
      isSearchTool(tool("agora_search_conversations", { query: { type: "string" } })),
    ).toBe(false);
    expect(
      isSearchTool(tool("search_conversations", { query: { type: "string" } })),
    ).toBe(false);
    expect(
      isSearchTool(tool("search_memories", { search_query: { type: "string" } })),
    ).toBe(false);
    expect(
      isSearchTool(
        {
          ...tool("lookup", { query: { type: "string" } }),
          function: {
            ...tool("lookup", { query: { type: "string" } }).function,
            description: "Search the web for current information.",
          },
        },
      ),
    ).toBe(true);
  });

  it("selects any advertised search tool for latest information", () => {
    const request = {
      model: "gpt-6-luna",
      messages: [
        { role: "user" as const, content: "Latest TikTok food hacks" },
      ],
      tools: [tool("web_lookup", { query: { type: "string" } })],
    };
    expect(
      createSearchToolCall("Latest TikTok food hacks", request.tools)?.function,
    ).toEqual({
      name: "web_lookup",
      arguments: '{"query":"Latest TikTok food hacks"}',
    });
  });

  it("preserves unrelated tools and gates a url fetch only when search is available", () => {
    const search = tool("web-browsing", { query: { type: "string" } });
    const fetch = tool("web-scraping", { url: { type: "string" } });
    const database = tool("database", { query: { type: "string" } });
    expect(isFetchTool(fetch)).toBe(true);
    expect(
      prioritizeSearchTools([search, fetch, database], false, "search the web"),
    ).toEqual([search, database]);
    expect(
      prioritizeSearchTools(
        [search, fetch, database],
        false,
        "read https://example.org",
      ),
    ).toEqual([search, fetch, database]);
    expect(
      prioritizeSearchTools(
        [search, fetch, database],
        false,
        "Create a document",
      ),
    ).toEqual([search, fetch, database]);
  });
});

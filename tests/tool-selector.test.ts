import { describe, expect, it } from "vitest";
import {
  isSearchTool,
  isFetchTool,
  prioritizeSearchTools,
  createSearchToolCall,
} from "../src/utils/tool-selector";
import type { OpenAITool } from "../src/utils/tool-shim";

function tool(name: string, properties: Record<string, unknown>): OpenAITool {
  return {
    type: "function",
    function: { name, parameters: { type: "object", properties } },
  };
}

describe("web tool classification", () => {
  it.each([
    ["web-browsing", "query"],
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

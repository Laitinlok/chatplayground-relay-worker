import { describe, expect, it } from "vitest";
import {
  buildToolSystemPrompt,
  gateToolCalls,
  injectReasoningPrompt,
  injectToolPrompt,
  normalizeOpenAITools,
  stripToolCallMarkup,
  toolCallRetrySignal,
  tryParseRelayToolCall,
  type OpenAITool,
} from "../src/utils/tool-shim";

const tools: OpenAITool[] = [
  { type: "function", function: { name: "cron" } },
  { type: "function", function: { name: "web_search" } },
];

describe("tool prompt updates", () => {
  it("replaces the relay policy and tool catalog while preserving caller instructions", () => {
    const first = injectToolPrompt(
      [{ role: "system", content: "Be concise." }],
      [tools[1]!],
      {
        type: "function",
        function: { name: "web_search" },
      },
    );
    const next = injectToolPrompt(first, tools, "required");
    const prompt = String(next[0]?.content);
    expect(prompt).toContain("Be concise.");
    expect(prompt).toContain("You must call exactly one listed tool.");
    expect(prompt).toContain("## cron");
    expect(prompt).not.toContain('You must call the tool named "web_search".');
    expect(prompt.match(/\[relay-tool-prompt-v1\]/g)).toHaveLength(1);
  });
});

describe("tryParseRelayToolCall", () => {
  it("maps a generic search call to the one advertised search tool and its query parameter", () => {
    const advertised: OpenAITool[] = [
      {
        type: "function",
        function: {
          name: "web_browse",
          parameters: {
            type: "object",
            properties: { search_query: { type: "string" } },
            required: ["search_query"],
          },
        },
      },
    ];
    const result = gateToolCalls(
      '<TOOL_CALL>\ntool: web_search\nparams: {"query":"latest recipes"}\n</TOOL_CALL>',
      advertised,
    );
    expect(result.rejections).toEqual([]);
    expect(result.calls[0]?.function).toEqual({
      name: "web_browse",
      arguments: '{"search_query":"latest recipes"}',
    });
  });

  it("recognizes search_news but excludes conversation search from web-search mapping", () => {
 const searchNews: OpenAITool[] = [
 {
 type: "function",
 function: {
 name: "search_news",
 parameters: {
 type: "object",
 properties: { query: { type: "string" } },
 },
 },
 },
 ];
 const genericSearchCall = `<TOOL_CALL>
tool: web_search
params: {"query":"latest news"}
</TOOL_CALL>`;
 expect(
 gateToolCalls(genericSearchCall, searchNews).calls[0]?.function.name,
 ).toBe("search_news");
 const conversationSearch: OpenAITool[] = [
 {
 type: "function",
 function: {
 name: "agora_search_conversations",
 parameters: {
 type: "object",
 properties: { query: { type: "string" } },
 },
 },
 },
 ];
 expect(gateToolCalls(genericSearchCall, conversationSearch).calls).toEqual(
 [],
 );
 });

 it("rejects a generic search call when multiple advertised search tools are ambiguous", () => {
    const advertised: OpenAITool[] = ["web_browse", "search_news"].map(
      (name) => ({
        type: "function",
        function: {
          name,
          parameters: {
            type: "object",
            properties: { query: { type: "string" } },
          },
        },
      }),
    );
    const result = gateToolCalls(
      '<TOOL_CALL>\ntool: web_search\nparams: {"query":"news"}\n</TOOL_CALL>',
      advertised,
    );
    expect(result.calls).toEqual([]);
    expect(result.rejections[0]).toContain("Unknown tool");
  });
  it("parses params JSON that contains braces inside string values", () => {
    const tools: OpenAITool[] = [
      {
        type: "function",
        function: {
          name: "edit_existing_file",
          parameters: {
            type: "object",
            properties: {
              filepath: { type: "string" },
              changes: { type: "string" },
            },
            required: ["filepath", "changes"],
          },
        },
      },
    ];
    const call = tryParseRelayToolCall(
      `<TOOL_CALL>
tool: edit_existing_file
params:
{"filepath":"a.js","changes":"if (x) { return 1; }"}
</TOOL_CALL>`,
      tools,
    );
    expect(call?.function.name).toBe("edit_existing_file");
    expect(JSON.parse(call!.function.arguments)).toEqual({
      filepath: "a.js",
      changes: "if (x) { return 1; }",
    });
  });

  it("rejects incomplete edit_existing_file calls with broken JSON params", () => {
    const tools: OpenAITool[] = [
      {
        type: "function",
        function: {
          name: "edit_existing_file",
          parameters: {
            type: "object",
            properties: {
              filepath: { type: "string" },
              changes: { type: "string" },
            },
            required: ["filepath", "changes"],
          },
        },
      },
    ];
    const broken = `<TOOL_CALL> tool: edit_existing_file params: {"filepath":"utils/searchBrave.js","changes":"function parseBraveData(source) {\n if (source.startsWith("[")) {\n let output = "";"}`;
    const gated = gateToolCalls(broken, tools);
    expect(gated.calls).toEqual([]);
    expect(gated.rejections.length).toBeGreaterThan(0);
    expect(gated.rejections.join(" ")).toMatch(/Incomplete|JSON|escaped/i);
    expect(stripToolCallMarkup(broken)).toBe("");
    expect(toolCallRetrySignal(gated.rejections)).toContain("</TOOL_CALL>");
  });

  it("rejects unknown tools and missing required args in the gate", () => {
    const tools: OpenAITool[] = [
      {
        type: "function",
        function: {
          name: "web_search",
          parameters: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
          },
        },
      },
    ];
    const unknown = gateToolCalls(
      "<TOOL_CALL>\ntool: not_a_tool\nparams:\n{}\n</TOOL_CALL>",
      tools,
    );
    expect(unknown.calls).toEqual([]);
    expect(unknown.rejections[0]).toContain("Unknown tool");

    const missing = gateToolCalls(
      "<TOOL_CALL>\ntool: web_search\nparams:\n{}\n</TOOL_CALL>",
      tools,
    );
    expect(missing.calls).toEqual([]);
    expect(missing.rejections[0]).toContain("missing required");
    expect(toolCallRetrySignal(missing.rejections)).toContain(
      "rejected by the tool gate",
    );
  });

  it("keeps valid calls and drops only the invalid ones", () => {
    const tools: OpenAITool[] = [
      {
        type: "function",
        function: {
          name: "cron",
          parameters: { type: "object", properties: {} },
        },
      },
      {
        type: "function",
        function: {
          name: "web_search",
          parameters: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
          },
        },
      },
    ];
    const gated = gateToolCalls(
      `<TOOL_CALL>
tool: cron
params:
{}
</TOOL_CALL>

<TOOL_CALL>
tool: web_search
params:
{}
</TOOL_CALL>`,
      tools,
    );
    expect(gated.calls.map((call) => call.function.name)).toEqual(["cron"]);
    expect(
      gated.rejections.some((reason) => reason.includes("web_search")),
    ).toBe(true);
  });

  it("parses the xml tool block", () => {
    const call = tryParseRelayToolCall(
      '<TOOL_CALL>\ntool: web_search\nparams:\n{"query":"latest news","max_results":3}\n</TOOL_CALL>',
      tools,
    );
    expect(call?.function.name).toBe("web_search");
    expect(call?.function.arguments).toContain("latest news");
  });

  it("accepts a close tool name inside the xml block", () => {
    const call = tryParseRelayToolCall(
      '<TOOL_CALL>\ntool: crno\nparams:\n{"action":"add"}\n</TOOL_CALL>',
      tools,
    );
    expect(call?.function.name).toBe("cron");
    expect(call?.function.arguments).toBe('{"action":"add"}');
  });

  it("ignores the old text protocol", () => {
    expect(
      tryParseRelayToolCall(
        'TOOL_CALL: web_search\nARGUMENTS: {"query":"food"}',
        tools,
      ),
    ).toBeNull();
  });

  it("rejects a tool that is not listed", () => {
    expect(
      tryParseRelayToolCall(
        "<TOOL_CALL>\ntool: unknown\nparams:\n{}\n</TOOL_CALL>",
        tools,
      ),
    ).toBeNull();
  });

  it("normalizes flat Responses-style tools for chat requests", () => {
    expect(
      normalizeOpenAITools([
        { type: "function", name: "ddg_search_search", parameters: {} },
      ]),
    ).toEqual([
      {
        type: "function",
        function: { name: "ddg_search_search", parameters: {} },
      },
    ]);
  });

  it("preserves web_search for Chat Completions normalization", () => {
    expect(
      normalizeOpenAITools([{ type: "web_search_preview" }]),
    ).toMatchObject([
      { type: "function", function: { name: "web_search" } },
      { type: "function", function: { name: "web_fetch" } },
    ]);
  });

  it("preserves web_search_preview only for Responses normalization", () => {
    expect(
      normalizeOpenAITools([{ type: "web_search_preview" }], {
        preserveWebSearchPreview: true,
      }),
    ).toMatchObject([
      { type: "function", function: { name: "web_search_preview" } },
      { type: "function", function: { name: "web_fetch" } },
    ]);
  });

  it("injects reasoning instructions with the requested strength", () => {
    const messages = injectReasoningPrompt(
      [{ role: "user", content: "solve this" }],
      "high",
    );
    expect(messages[0]).toMatchObject({ role: "system" });
    expect(messages[0]?.content).toContain("[relay-reasoning-prompt-v1]");
    expect(messages[0]?.content).toContain('strength "high"');
    expect(messages[0]?.content).toContain("<think>...</think>");
  });

  it("does not duplicate reasoning instructions", () => {
    const first = injectReasoningPrompt(
      [{ role: "user", content: "solve this" }],
      "medium",
    );
    expect(injectReasoningPrompt(first, "high")).toEqual(first);
  });

  it("strips image-generation wording and retains the subject for prompt refinement", () => {
    const messages = injectToolPrompt(
      [{ role: "user", content: "Please generate an image of a large cafe." }],
      [
        {
          type: "function",
          function: { name: "generate_image" },
        },
      ],
      "auto",
    );
    expect(messages.at(-1)?.content).toBe("a large cafe");
    expect(messages[0]?.content).toContain(
      "improve it into a polished, descriptive image prompt",
    );
    expect(messages[0]?.content).toContain(
      "do not answer with a refusal or claim image generation is unavailable",
    );
  });

  it("does not rewrite image requests when generate_image is not advertised", () => {
    const messages = injectToolPrompt(
      [{ role: "user", content: "Please generate an image of a large cafe." }],
      [{ type: "function", function: { name: "create_document" } }],
      "auto",
    );
    expect(messages.at(-1)?.content).toBe(
      "Please generate an image of a large cafe.",
    );
  });

  it("injects a compact catalog with required parameter guidance", () => {
    const prompt = buildToolSystemPrompt([
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get current weather",
          parameters: {
            type: "object",
            properties: { city: { type: "string", description: "City name" } },
            required: ["city"],
          },
        },
      },
    ]);
    expect(prompt).toContain("<TOOL_CALL>");
    expect(prompt).toContain("get_weather");
    expect(prompt).toContain('"city"');
    expect(prompt).toContain("<AVAILABLE_TOOLS>");
    expect(prompt).toContain("one <TOOL_CALL> block per tool");
  });
  it("adds a strict dispatch protocol for Perplexity models", () => {
    const prompt = buildToolSystemPrompt(
      [{ type: "function", function: { name: "edit_file" } }],
      "auto",
      "perplexity",
    );
    expect(prompt).toContain("<TOOL_RULES>");
    expect(prompt).toContain("edit_file");
    expect(prompt).toContain("<TOOL_CALL>");
  });
});

import { describe, expect, it } from "vitest";
import {
  buildToolSystemPrompt,
  gateToolCalls,
  injectReasoningPrompt,
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

describe("tryParseRelayToolCall", () => {
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
    expect(toolCallRetrySignal(missing.rejections)).toContain("rejected by the tool gate");
  });

  it("keeps valid calls and drops only the invalid ones", () => {
    const tools: OpenAITool[] = [
      { type: "function", function: { name: "cron", parameters: { type: "object", properties: {} } } },
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
    expect(gated.rejections.some((reason) => reason.includes("web_search"))).toBe(true);
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
        '<TOOL_CALL>\ntool: unknown\nparams:\n{}\n</TOOL_CALL>',
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

  it("normalizes OpenAI built-in web search for the relay tool shim", () => {
    expect(
      normalizeOpenAITools([{ type: "web_search_preview" }]),
    ).toMatchObject([
      { type: "function", function: { name: "web_search" } },
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



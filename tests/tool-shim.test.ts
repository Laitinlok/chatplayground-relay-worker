import { describe, expect, it } from "vitest";
import {
  buildToolSystemPrompt,
  injectReasoningPrompt,
  normalizeOpenAITools,
  tryParseRelayToolCall,
  type OpenAITool,
} from "../src/utils/tool-shim";

const tools: OpenAITool[] = [
  { type: "function", function: { name: "cron" } },
  { type: "function", function: { name: "web_search" } },
];

describe("tryParseRelayToolCall", () => {
  it("accepts close envelope and tool names", () => {
    const call = tryParseRelayToolCall(
      '{"relayToolCall":{"toolName":"crno","params":{"action":"add"}}}',
      tools,
    );
    expect(call?.function.name).toBe("cron");
    expect(call?.function.arguments).toBe('{"action":"add"}');
  });

  it("accepts alternate envelope keys and JSON-encoded arguments", () => {
    const call = tryParseRelayToolCall(
      '{"function-call":{"function-name":"web_search","args":"{\\"query\\":\\"openclaw\\"}"}}',
      tools,
    );
    expect(call?.function.name).toBe("web_search");
    expect(call?.function.arguments).toBe('{"query":"openclaw"}');
  });

  it("rejects ambiguous fuzzy matches", () => {
    const ambiguous: OpenAITool[] = [
      { type: "function", function: { name: "calendar" } },
      { type: "function", function: { name: "calender" } },
    ];
    expect(
      tryParseRelayToolCall(
        '{"call":{"name":"calendr","arguments":{}}}',
        ambiguous,
      ),
    ).toBeNull();
  });

  it("parses the injected TOOL_CALL text protocol", () => {
    const call = tryParseRelayToolCall(
      'I will check that.\nTOOL_CALL: crno\nARGUMENTS: {"action":"add","amount":6600}',
      tools,
    );
    expect(call?.function.name).toBe("cron");
    expect(call?.function.arguments).toBe('{"action":"add","amount":6600}');
  });

  it("parses GPT-5 Harmony recipient tool calls", () => {
    const call = tryParseRelayToolCall(
      '<|channel|>commentary to=functions.web_search <|constrain|>json<|message|>{"query":"latest news"}',
      tools,
    );
    expect(call?.function.name).toBe("web_search");
    expect(call?.function.arguments).toBe('{"query":"latest news"}');
  });

  it("does not expose an unknown Harmony recipient as a tool call", () => {
    expect(
      tryParseRelayToolCall(
        '<|channel|>commentary to=functions.delete_all <|message|>{"confirm":true}',
        tools,
      ),
    ).toBeNull();
  });

  it("accepts inline TOOL_CALL labels", () => {
    const call = tryParseRelayToolCall(
      'I will search. TOOL_CALL: web_search ARGUMENTS: {"query":"food hacks"}',
      tools,
    );
    expect(call?.function.name).toBe("web_search");
    expect(call?.function.arguments).toBe('{"query":"food hacks"}');
  });

  it("parses the exact relay text emitted by non-5.x model families", () => {
    const call = tryParseRelayToolCall(
      'I\'ll search the web to find the latest TikTok food hacks for you.\r\nTOOL_CALL: ddg_search_search\r\nARGUMENTS: {"query":"latest TikTok food hacks 2025","max_results":10}',
      normalizeOpenAITools([
        {
          type: "function",
          name: "ddg_search_search",
          description: "Search the web",
          parameters: { type: "object" },
        },
      ]),
    );
    expect(call?.function.name).toBe("ddg_search_search");
    expect(call?.function.arguments).toBe(
      '{"query":"latest TikTok food hacks 2025","max_results":10}',
    );
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

  it("forces Luna to emit a tool call for default auto choice", () => {
    const prompt = buildToolSystemPrompt(tools, undefined, "gpt-5.6-luna");
    expect(prompt).toContain("Do not answer directly");
  });

  it("keeps Luna tool choice none disabled", () => {
    const prompt = buildToolSystemPrompt(tools, "none", "gpt-5.6-luna");
    expect(prompt).toContain("must not call any tool");
  });  it("does not duplicate reasoning instructions", () => {
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
    expect(prompt).toContain("TOOL_CALL: <tool name>");
    expect(prompt).toContain("city (required): City name");
    expect(prompt).toContain("Only use tools from this list");
    expect(prompt).toContain("complete, self-contained final answer");
    expect(prompt).toContain("emit the call immediately");
  });
  it("parses the first call once when the model duplicates the payload", () => {
    const payload =
      '{"relay_tool_call":{"name":"cron","arguments":{"action":"add"}}}';
    const call = tryParseRelayToolCall(`${payload}${payload}`, tools);
    expect(call?.function.name).toBe("cron");
    expect(call?.function.arguments).toBe('{"action":"add"}');
  });
});

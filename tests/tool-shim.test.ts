import { describe, expect, it } from "vitest";
import {
  buildToolSystemPrompt,
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
      tryParseRelayToolCall('{"call":{"name":"calendr","arguments":{}}}', ambiguous),
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
  });
  it("parses the first call once when the model duplicates the payload", () => {
    const payload = '{"relay_tool_call":{"name":"cron","arguments":{"action":"add"}}}';
    const call = tryParseRelayToolCall(`${payload}${payload}`, tools);
    expect(call?.function.name).toBe("cron");
    expect(call?.function.arguments).toBe('{"action":"add"}');
  });
});

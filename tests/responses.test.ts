import { describe, expect, it } from "vitest";
import {
  buildToolSystemPrompt,
  injectToolPrompt,
} from "../src/utils/tool-shim";
import type { OpenAITool } from "../src/utils/tool-shim";
import {
  chatResultToResponses,
  responsesToChatRequest,
  responsesToolsToChatTools,
} from "../src/types/responses";

describe("Responses adapter", () => {
	it("maps GPT-5.6-style input and function tools to chat shapes", () => {
		const request = responsesToChatRequest({
			model: "gpt-5.6",
			instructions: "Be concise.",
			input: "Find it",
			tools: [
				{
					type: "function",
					name: "web_search",
					parameters: { type: "object" },
				},
			],
			tool_choice: "required",
		});

		expect(request.messages).toEqual([
		  { role: "system", content: "Be concise." },
		  { role: "user", content: "Find it" },
		]);
		expect(request.tools?.[0]?.function.name).toBe("web_search");
		expect(request.tool_choice).toBe("required");
	});

	it("preserves function call and function call output history", () => {
		const request = responsesToChatRequest({
			model: "gpt-5.6",
			input: [
				{
					type: "function_call",
					call_id: "call_123",
					name: "cron",
					arguments: '{"action":"add"}',
				},
				{
					type: "function_call_output",
					call_id: "call_123",
					output: "created",
				},
			],
		});

		expect(request.messages[0]?.tool_calls?.[0]?.id).toBe("call_123");
		expect(request.messages[1]).toMatchObject({
			role: "tool",
			content: "created",
			name: "call_123",
		});
	});

	it("serializes text and tool calls as Responses output items", () => {
		const text = chatResultToResponses(
			"gpt-5.6",
			"hello",
			null,
			[{ role: "user", content: "hi" }],
			"resp_test",
		);
		expect(text.object).toBe("response");
		expect(text.output_text).toBe("hello");
		expect(text.output[0]).toMatchObject({
			type: "message",
			role: "assistant",
		});

		const call = chatResultToResponses(
			"gpt-5.6",
			"",
			{
				id: "call_123",
				type: "function",
				function: { name: "cron", arguments: '{"action":"add"}' },
			},
			[],
			"resp_tool",
		);
		expect(call.output[0]).toMatchObject({
			type: "function_call",
			call_id: "call_123",
			name: "cron",
			arguments: '{"action":"add"}',
		});
		expect(call.output_text).toBe("");
	});

  it("does not duplicate the injected tool prompt", () => {
    const tool: OpenAITool[] = [{ type: "function", function: { name: "cron" } }];
    const first = injectToolPrompt([{ role: "user", content: "hi" }], tool);
    const second = injectToolPrompt(first, tool);
    expect(second).toEqual(first);
    expect(buildToolSystemPrompt(tool)).toContain("[relay-tool-prompt-v1]");
  });

  it("accepts the nested function tool form for compatibility", () => {
    expect(
      responsesToolsToChatTools([
        { type: "function", function: { name: "cron" } },
      ]),
    ).toEqual([{ type: "function", function: { name: "cron" } }]);
  });
});

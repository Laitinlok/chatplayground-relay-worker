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
  type ResponsesResponse,
} from "../src/types/responses";
import { streamErrorResponse, streamResponse } from "../src/routes/responses";
import {
  formatSearchResults,
  formatSearchSources,
  normalizeSearchLimit,
} from "../src/utils/web-search";

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

  it("maps native OpenAI Search to the relay search shim", () => {
    const request = responsesToChatRequest({
      model: "gpt-5.6",
      input: "Find current information",
      tools: [{ type: "web_search" }],
    });

    expect(request.tools?.[0]).toMatchObject({
      type: "function",
      function: { name: "web_search" },
    });
  });

  it("lets the model choose a bounded number of native search results", () => {
    const tool = responsesToolsToChatTools([{ type: "web_search" }])?.[0];
    const parameters = tool?.function.parameters as {
      properties: { num_results: Record<string, unknown> };
      required?: string[];
    };

    expect(parameters.properties.num_results).toMatchObject({
      type: "integer",
      minimum: 1,
      maximum: 10,
    });
    expect(parameters.properties.num_results.description).toContain(
      "smallest useful number",
    );
    expect(parameters.required).toBeUndefined();
    expect(normalizeSearchLimit(7)).toBe(7);
    expect(normalizeSearchLimit(100)).toBe(10);
    expect(normalizeSearchLimit(Number.NaN)).toBe(5);
  });
  it("formats native search results as sources without echoing the query", () => {
    const results = [
      {
        title: "Example source",
        url: "https://example.com/article",
        snippet: "Useful evidence.",
      },
    ];
    const content = formatSearchResults(results) + formatSearchSources(results);

    expect(content).toContain("1. Example source\nUseful evidence.");
    expect(content).toContain("**Sources**");
    expect(content).toContain("[Example source](https://example.com/article)");
    expect(content).not.toContain("Search results for");
  });
  it("accepts a single OmniRoute input item object", () => {
    const request = responsesToChatRequest({
      model: "gpt-5.6",
      input: { role: "user", content: "Find it" },
    });

    expect(request.messages).toEqual([{ role: "user", content: "Find it" }]);
  });
  it("uses messages as a compatibility fallback", () => {
    const request = responsesToChatRequest({
      model: "gpt-5.6",
      messages: [{ role: "user", content: "Find it" }],
    });

    expect(request.messages).toEqual([{ role: "user", content: "Find it" }]);
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

  it("serializes a native web search call output", () => {
    const response = chatResultToResponses(
      "gpt-5.6",
      "",
      {
        id: "call_search",
        type: "function",
        function: {
          name: "web_search",
          arguments: '{"query":"latest news"}',
        },
      },
      [],
      "resp_search",
      "",
      true,
    );

    expect(response.output[0]).toEqual({
      type: "web_search_call",
      id: "ws_search",
      status: "completed",
      action: { type: "search", query: "latest news" },
    });
    expect(response.output).toHaveLength(1);
  });

  it("keeps search citations clickable in the synthesized response", () => {
    const response = chatResultToResponses(
      "gpt-5.6",
      "The answer is supported by [\\[1\\]](https://example.com/article).\n\n---\n**Sources**\n\n1. [Example source](https://example.com/article)",
      {
        id: "call_search",
        type: "function",
        function: {
          name: "web_search",
          arguments: '{"query":"latest news","num_results":1}',
        },
      },
      [],
      "resp_search_citations",
      "",
      true,
      [
        {
          title: "Example source",
          url: "https://example.com/article",
          snippet: "Useful evidence.",
        },
      ],
    );

    expect(response.output_text).toContain(
      "[\\[1\\]](https://example.com/article)",
    );
    expect(response.output[0]).toMatchObject({
      type: "web_search_call",
      action: { results: [{ url: "https://example.com/article" }] },
    });
  });

  it("emits Responses-compatible envelopes for streaming clients", async () => {
    const result = chatResultToResponses(
      "gpt-5.6",
      "hello",
      null,
      [{ role: "user", content: "hi" }],
      "resp_stream",
    ) as ResponsesResponse;
    const body = await new Response(streamResponse(result, null)).text();

    expect(body).toContain(
      'event: response.created\ndata: {"type":"response.created","response":',
    );
    expect(body).toContain(
      'event: response.completed\ndata: {"type":"response.completed","response":',
    );
    const payloads = body
      .split("\n")
      .filter((line) => line.startsWith("data: "))
      .map((line) => JSON.parse(line.slice(6)) as Record<string, unknown>);
    expect(payloads.length).toBeGreaterThan(0);
    let previous = -1;
    for (const payload of payloads) {
      expect(payload.type).toBeDefined();
      expect(payload.sequence_number).toBeGreaterThan(previous);
      previous = payload.sequence_number as number;
    }
  });
  it("terminates streamed upstream failures with response.failed", async () => {
    const response = streamErrorResponse(
      "gpt-5.6",
      429,
      "Upstream returned 429: rate limited",
    );
    const body = await response.text();

    expect(response.status).toBe(429);
    expect(body).toContain("event: response.failed");
    expect(body).toContain('"status":"failed"');
    expect(body).toContain('"type":"rate_limit_error"');
    expect(body).toContain("event: error");
  });
  it("normalizes upstream 500 failures to HTTP 502 for OmniRoute", async () => {
    const response = streamErrorResponse(
      "gpt-5.6",
      500,
      "Upstream returned 500: provider failed",
    );

    expect(response.status).toBe(502);
    expect(await response.text()).toContain('"code":"upstream_500"');
  });
  it("streams a native web search output item", async () => {
    const result = chatResultToResponses(
      "gpt-5.6",
      "",
      {
        id: "call_search",
        type: "function",
        function: {
          name: "web_search",
          arguments: '{"query":"latest news"}',
        },
      },
      [],
      "resp_search_stream",
      "",
      true,
    );
    const body = await new Response(streamResponse(result, null)).text();

    expect(body).toContain("event: response.output_item.added");
    expect(body).toContain("response.web_search_call.in_progress");
    expect(body).toContain("response.web_search_call.searching");
    expect(body).toContain("response.web_search_call.completed");
    expect(body).toContain('"action":{"type":"search","query":"latest news"}');
  });

  it("does not duplicate the injected tool prompt", () => {
    const tool: OpenAITool[] = [
      { type: "function", function: { name: "cron" } },
    ];
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

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
import { addHostedWebSearch, streamResponse } from "../src/routes/responses";

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
    expect(body).toContain('event: response.completed\ndata: {"type":"response.completed","response":');
    expect(body).toContain(
      '"choices":[{"index":0,"delta":{"content":"hello"}}]',
    );
    const payloads = body
      .split("\n")
      .filter((line) => line.startsWith("data: "))
      .map((line) => JSON.parse(line.slice(6)) as Record<string, unknown>);
    expect(payloads.length).toBeGreaterThan(0);
    for (const payload of payloads) {
      expect(typeof payload.sequence_number).toBe("number");
      expect(typeof payload.type).toBe("string");
    }
  });
  it("emits the hosted web-search lifecycle before the answer", async () => {
    const result = chatResultToResponses(
      "gpt-5.6",
      "hello",
      null,
      [{ role: "user", content: "hi" }],
      "resp_search",
    );
    result.output.unshift({
      type: "web_search_call",
      id: "ws_1",
      status: "completed",
      action: { type: "search", query: "latest news" },
      results: [
        {
          title: "Latest news",
          url: "https://news.example/article",
          snippet: "A current report.",
        },
      ],
    });
    const body = await new Response(streamResponse(result, null)).text();
    expect(body).toContain("event: response.web_search_call.in_progress");
    expect(body).toContain("event: response.web_search_call.searching");
    expect(body).toContain("event: response.web_search_call.completed");
    expect(body).toContain('"action":{"type":"search","query":"latest news"}');
  });

  it("returns sources as standard url_citation annotations", () => {
    const base = chatResultToResponses(
      "gpt-5.6",
      "Here is the answer.",
      null,
      [{ role: "user", content: "hi" }],
      "resp_citations",
    );
    const result = addHostedWebSearch(base, "latest news", [
      {
        title: "Latest news",
        url: "https://news.example/article",
        snippet: "A current report.",
      },
    ]);
    const webCall = result.output[0];
    const message = result.output[1];

    expect(webCall).toMatchObject({
      type: "web_search_call",
      action: { type: "search", query: "latest news" },
    });
    expect(message).toMatchObject({
      type: "message",
      content: [
        {
          text: "Here is the answer.\n\nSources:\n1. Latest news",
          annotations: [
            {
              type: "url_citation",
              url: "https://news.example/article",
              title: "Latest news",
            },
          ],
        },
      ],
    });
    expect(result.output_text).toBe("Here is the answer.\n\nSources:\n1. Latest news");
  });

  it("keeps sequence numbers contiguous for Responses clients", async () => {
    const result = chatResultToResponses(
      "gpt-5.6",
      "hello",
      null,
      [{ role: "user", content: "hi" }],
      "resp_omni",
    );
    const body = await new Response(streamResponse(result, null)).text();
    const sequenceNumbers = body
      .split("\n")
      .filter((line) => line.startsWith("data: "))
      .map(
        (line) =>
          (JSON.parse(line.slice(6)) as Record<string, unknown>)
            .sequence_number,
      );
    expect(sequenceNumbers).toEqual(
      sequenceNumbers.map((_value, index) => index),
    );
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

  it("maps OpenAI web search tools into the relay search tool", () => {
    expect(
      responsesToolsToChatTools([
        { type: "web_search_preview", search_context_size: "high" },
      ]),
    ).toEqual([
      expect.objectContaining({
        type: "function",
        function: expect.objectContaining({ name: "web_search" }),
      }),
      expect.objectContaining({
        type: "function",
        function: expect.objectContaining({ name: "web_fetch" }),
      }),
    ]);
  });

  it("accepts the nested function tool form for compatibility", () => {
    expect(
      responsesToolsToChatTools([
        { type: "function", function: { name: "cron" } },
      ]),
    ).toEqual([{ type: "function", function: { name: "cron" } }]);
  });
});

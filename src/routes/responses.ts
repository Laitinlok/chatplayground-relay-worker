import { Hono } from "hono";
import type { ChatCompletionRequest } from "../types/openai";
import { CHAT_TIMEOUT } from "../constants/timeouts";
import type { Env, Variables } from "../types/env";

import type {
  ResponsesRequest,
  ResponsesResponse,
  ResponsesWebSearchResult,
} from "../types/responses";
import {
  chatResultToResponses,
  hasNativeWebSearch,
  responsesToChatRequest,
} from "../types/responses";
import { invalidRequest, modelNotFound, upstreamError } from "../utils/errors";
import { getModels } from "../utils/model-discovery";
import { findModel } from "../utils/model-id";
import {
  DEFAULT_MAX_RESULTS,
  formatSearchResults,
  formatSearchSources,
  normalizeSearchLimit,
  searchWeb,
} from "../utils/web-search";
import {
  buildUpstreamHeaders,
  buildUpstreamRequest,
  endpointUrl,
} from "../utils/upstream-request";
import {
  collectUpstream,
  formatCitations,
  inlineCitationLinks,
  splitReasoningContent,
} from "../utils/upstream-stream";
import {
  hasTools,
  injectReasoningPrompt,
  injectToolPrompt,
  tryParseRelayToolCall,
} from "../utils/tool-shim";

const responses = new Hono<{ Bindings: Env; Variables: Variables }>();

function event(type: string, data: unknown): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

export function streamErrorResponse(
  model: string,
  status: number,
  message: string,
): Response {
  const normalizedError = upstreamError(status, message);
  const response = {
    id: `resp_${crypto.randomUUID().replace(/-/g, "")}`,
    object: "response" as const,
    created_at: Math.floor(Date.now() / 1000),
    status: "failed" as const,
    model,
    output: [],
    output_text: "",
    error: {
      code: normalizedError.code,
      type: normalizedError.type,
      message: normalizedError.message,
    },
    usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
  };
  const body =
    event("response.failed", {
      type: "response.failed",
      response,
      sequence_number: 0,
    }) +
    event("error", {
      type: "error",
      error: response.error,
      sequence_number: 1,
    });
  return new Response(body, {
    status: normalizedError.status,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
    },
  });
}

export function streamResponse(
  result: ResponsesResponse,
  toolCall: ReturnType<typeof tryParseRelayToolCall>,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      let sequenceNumber = 0;
      const send = (type: string, data: unknown) =>
        event(type, {
          ...(data as Record<string, unknown>),
          sequence_number: sequenceNumber++,
        });
      const response = {
        ...result,
        status: "in_progress" as const,
        output: [],
        output_text: "",
      };
      controller.enqueue(
        encoder.encode(
          send("response.created", {
            type: "response.created",
            response,
          }),
        ),
      );
      controller.enqueue(
        encoder.encode(
          send("response.in_progress", {
            type: "response.in_progress",
            response,
          }),
        ),
      );
      const output = result.output[0];

      if (output?.type === "web_search_call") {
        const inProgressOutput = { ...output, status: "in_progress" };
        controller.enqueue(
          encoder.encode(
            send("response.output_item.added", {
              type: "response.output_item.added",
              output_index: 0,
              item: inProgressOutput,
            }),
          ),
        );
        for (const type of [
          "response.web_search_call.in_progress",
          "response.web_search_call.searching",
          "response.web_search_call.completed",
        ]) {
          controller.enqueue(
            encoder.encode(
              send(type, {
                type,
                output_index: 0,
                item_id: output.id,
              }),
            ),
          );
        }
        controller.enqueue(
          encoder.encode(
            send("response.output_item.done", {
              type: "response.output_item.done",
              output_index: 0,
              item: output,
            }),
          ),
        );
        const answer = result.output[1];
        if (answer?.type === "message") {
          const text = result.output_text;
          const part = answer.content[0];
          controller.enqueue(
            encoder.encode(
              send("response.output_item.added", {
                type: "response.output_item.added",
                output_index: 1,
                item: { ...answer, content: [] },
              }),
            ),
          );
          controller.enqueue(
            encoder.encode(
              send("response.content_part.added", {
                type: "response.content_part.added",
                item_id: answer.id,
                output_index: 1,
                content_index: 0,
                part,
              }),
            ),
          );
          if (text)
            controller.enqueue(
              encoder.encode(
                send("response.output_text.delta", {
                  type: "response.output_text.delta",
                  item_id: answer.id,
                  output_index: 1,
                  content_index: 0,
                  delta: text,
                }),
              ),
            );
          controller.enqueue(
            encoder.encode(
              send("response.output_text.done", {
                type: "response.output_text.done",
                item_id: answer.id,
                output_index: 1,
                content_index: 0,
                text,
              }),
            ),
          );
          controller.enqueue(
            encoder.encode(
              send("response.content_part.done", {
                type: "response.content_part.done",
                item_id: answer.id,
                output_index: 1,
                content_index: 0,
                part,
              }),
            ),
          );
          controller.enqueue(
            encoder.encode(
              send("response.output_item.done", {
                type: "response.output_item.done",
                output_index: 1,
                item: answer,
              }),
            ),
          );
        }
      } else if (toolCall && output?.type === "function_call") {
        controller.enqueue(
          encoder.encode(
            send("response.output_item.added", {
              type: "response.output_item.added",
              output_index: 0,
              item: output,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.function_call_arguments.delta", {
              type: "response.function_call_arguments.delta",
              item_id: output.id,
              output_index: 0,
              delta: output.arguments,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.function_call_arguments.done", {
              type: "response.function_call_arguments.done",
              item_id: output.id,
              output_index: 0,
              arguments: output.arguments,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.output_item.done", {
              type: "response.output_item.done",
              output_index: 0,
              item: output,
            }),
          ),
        );
      } else if (output?.type === "message") {
        const text = result.output_text;
        const part = output.content[0];
        controller.enqueue(
          encoder.encode(
            send("response.output_item.added", {
              type: "response.output_item.added",
              output_index: 0,
              item: { ...output, content: [] },
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.content_part.added", {
              type: "response.content_part.added",
              item_id: output.id,
              output_index: 0,
              content_index: 0,
              part,
            }),
          ),
        );
        if (text)
          controller.enqueue(
            encoder.encode(
              send("response.output_text.delta", {
                type: "response.output_text.delta",
                item_id: output.id,
                output_index: 0,
                content_index: 0,
                delta: text,
              }),
            ),
          );
        controller.enqueue(
          encoder.encode(
            send("response.output_text.done", {
              type: "response.output_text.done",
              item_id: output.id,
              output_index: 0,
              content_index: 0,
              text,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.content_part.done", {
              type: "response.content_part.done",
              item_id: output.id,
              output_index: 0,
              content_index: 0,
              part,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.output_item.done", {
              type: "response.output_item.done",
              output_index: 0,
              item: output,
            }),
          ),
        );
      }
      controller.enqueue(
        encoder.encode(
          send("response.completed", {
            type: "response.completed",
            response: result,
          }),
        ),
      );
      controller.close();
    },
  });
}

function removeToolPrompt(
  messages: ChatCompletionRequest["messages"],
): ChatCompletionRequest["messages"] {
  return messages.flatMap((message) => {
    if (message.role !== "system" || typeof message.content !== "string")
      return [message];
    const marker = "[relay-tool-prompt-v1]";
    const markerIndex = message.content.indexOf(marker);
    if (markerIndex < 0) return [message];
    const content = message.content.slice(0, markerIndex).trim();
    return content ? [{ ...message, content }] : [];
  });
}
function extractSearchQuery(argumentsJson: string): string {
  try {
    const value = JSON.parse(argumentsJson) as {
      query?: unknown;
      queries?: unknown;
    };
    if (typeof value.query === "string") return value.query;
    if (Array.isArray(value.queries))
      return value.queries.find((query) => typeof query === "string") ?? "";
  } catch {
    // The search result remains bounded even if the model emitted invalid JSON.
  }
  return "";
}

function extractSearchLimit(argumentsJson: string): number {
  try {
    const value = JSON.parse(argumentsJson) as { num_results?: unknown };
    return normalizeSearchLimit(value.num_results);
  } catch {
    return DEFAULT_MAX_RESULTS;
  }
}

async function buildNativeSearchAnswer(
  toolCall: ReturnType<typeof tryParseRelayToolCall>,
): Promise<{ content: string; results: ResponsesWebSearchResult[] }> {
  if (!toolCall) return { content: "", results: [] };
  const query = extractSearchQuery(toolCall.function.arguments);
  try {
    const results = await searchWeb(
      query,
      extractSearchLimit(toolCall.function.arguments),
    );
    return {
      content: formatSearchResults(results) + formatSearchSources(results),
      results,
    };
  } catch (error) {
    const detail = error instanceof Error ? error.message : "search failed";
    return {
      content: `Search unavailable: ${detail}`,
      results: [],
    };
  }
}

responses.post("/v1/responses", async (c) => {
  const raw = (await c.req.json().catch(() => null)) as ResponsesRequest | null;
  if (!raw || typeof raw !== "object")
    throw invalidRequest("Request body must be JSON.");
  if (!raw.model || typeof raw.model !== "string")
    throw invalidRequest("'model' is required.", "model");
  const input = raw.input ?? raw.messages ?? raw.prompt;
  if (
    typeof input !== "string" &&
    !Array.isArray(input) &&
    (!input || typeof input !== "object")
  )
    throw invalidRequest(
      "'input' must be a string, object, or array (or provide messages).",
      "input",
    );

  const registry = await getModels(c.env);
  const model = findModel(raw.model, registry);
  if (!model) throw modelNotFound(raw.model);

  const request = responsesToChatRequest({ ...raw, input });
  const reasoningEffort = request.reasoning_effort ?? "medium";
  request.reasoning_effort = reasoningEffort;
  request.messages = injectReasoningPrompt(request.messages, reasoningEffort);
  if (request.messages.length === 0)
    throw invalidRequest("'input' must contain at least one message.", "input");

  const tools = request.tools;
  const nativeWebSearch = hasNativeWebSearch(raw.tools);
  const toolsRequested = hasTools(tools);
  if (toolsRequested)
    request.messages = injectToolPrompt(
      request.messages,
      tools,
      request.tool_choice,
    );

  const sessionToken = await c.get("sessionToken")();
  const fetchUpstream = async (chatRequest: typeof request) => {
    const { endpoint, body } = buildUpstreamRequest(chatRequest, model);
    return fetch(endpointUrl(endpoint, c.env.UPSTREAM_CHAT_URL), {
      method: "POST",
      headers: buildUpstreamHeaders(sessionToken, c.env),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CHAT_TIMEOUT),
    });
  };
  const upstream = await fetchUpstream(request);
  if (!upstream.ok || !upstream.body) {
    const detail = (await upstream.text().catch(() => "")).trim();
    const message = detail
      ? `Upstream returned ${upstream.status}: ${detail.slice(0, 300)}`
      : `Upstream returned ${upstream.status} with no message.`;
    if (raw.stream) {
      return streamErrorResponse(raw.model, upstream.status, message);
    }
    throw upstreamError(upstream.status, message);
  }

  const parsed = await collectUpstream(upstream.body);
  if (!parsed.content.trim() && parsed.citations.length === 0) {
    const message = "Provider returned HTTP 200 but no text content.";
    if (raw.stream) return streamErrorResponse(raw.model, 200, message);
    throw upstreamError(200, message);
  }
  const rawContent = parsed.content;
  const toolCall = toolsRequested
    ? tryParseRelayToolCall(rawContent, request.tools)
    : null;
  const split = splitReasoningContent(rawContent);
  const visibleContent = split.content || split.reasoningContent;
  const searchAnswer =
    toolCall && nativeWebSearch
      ? await buildNativeSearchAnswer(toolCall)
      : { content: "", results: [] };

  let answerContent = visibleContent;
  let answerReasoning = split.reasoningContent;
  let answerCitations = parsed.citations;
  if (toolCall && nativeWebSearch) {
    const synthesisRequest = {
      ...request,
      tools: undefined,
      tool_choice: undefined,
      messages: [
        ...removeToolPrompt(request.messages),
        {
          role: "assistant" as const,
          content: null,
          tool_calls: [toolCall],
        },
        {
          role: "tool" as const,
          name: toolCall.function.name,
          content: searchAnswer.content,
        },
        {
          role: "system" as const,
          content:
            "The search tool result above is complete. Answer the user's original request directly using the supplied evidence. Do not emit any tool call, mention the tool protocol, or output a separate sources list; the relay will add numbered clickable citations.",
        },
      ],
    };
    const synthesisUpstream = await fetchUpstream(synthesisRequest);
    if (!synthesisUpstream.ok || !synthesisUpstream.body) {
      const detail = (await synthesisUpstream.text().catch(() => "")).trim();
      const message = detail
        ? `Upstream returned ${synthesisUpstream.status}: ${detail.slice(0, 300)}`
        : `Upstream returned ${synthesisUpstream.status} with no message.`;
      if (raw.stream)
        return streamErrorResponse(
          raw.model,
          synthesisUpstream.status,
          message,
        );
      throw upstreamError(synthesisUpstream.status, message);
    }
    const synthesized = await collectUpstream(synthesisUpstream.body);
    if (!synthesized.content.trim() && synthesized.citations.length === 0) {
      const message = "Provider returned HTTP 200 but no text content.";
      if (raw.stream) return streamErrorResponse(raw.model, 200, message);
      throw upstreamError(200, message);
    }
    const synthesizedSplit = splitReasoningContent(synthesized.content);
    answerContent =
      synthesizedSplit.content || synthesizedSplit.reasoningContent;
    answerReasoning = synthesizedSplit.reasoningContent;
    answerCitations = synthesized.citations;
  }
  const content =
    toolCall && nativeWebSearch
      ? inlineCitationLinks(
          answerContent,
          searchAnswer.results.map((result) => result.url),
        ) + formatSearchSources(searchAnswer.results)
      : toolCall
        ? ""
        : answerCitations.length
          ? inlineCitationLinks(answerContent, answerCitations) +
            formatCitations(answerCitations)
          : answerContent;
  const result = chatResultToResponses(
    model.id,
    content,
    toolCall,
    request.messages,
    undefined,
    answerReasoning,
    nativeWebSearch,
    searchAnswer.results,
  );

  if (raw.stream) {
    return new Response(streamResponse(result, toolCall), {
      headers: {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
      },
    });
  }
  return Response.json(result);
});

export default responses;

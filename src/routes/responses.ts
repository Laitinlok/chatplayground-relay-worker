import { Hono } from "hono";
import { CHAT_TIMEOUT } from "../constants/timeouts";
import type { Env, Variables } from "../types/env";
import type { ResponsesRequest, ResponsesResponse } from "../types/responses";
import {
  chatResultToResponses,
  responsesToChatRequest,
} from "../types/responses";
import { invalidRequest, modelNotFound, upstreamError } from "../utils/errors";
import { getModels } from "../utils/model-discovery";
import { findModel } from "../utils/model-id";
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

export function streamResponse(
  result: ResponsesResponse,
  toolCall: ReturnType<typeof tryParseRelayToolCall>,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      const response = { ...result, status: "in_progress" };
      controller.enqueue(
        encoder.encode(
          event("response.created", {
            type: "response.created",
            response,
          }),
        ),
      );
      controller.enqueue(
        encoder.encode(
          event("response.in_progress", {
            type: "response.in_progress",
            response,
          }),
        ),
      );
      const output = result.output[0];
      if (toolCall && output?.type === "function_call") {
        controller.enqueue(
          encoder.encode(
            event("response.output_item.added", {
              type: "response.output_item.added",
              output_index: 0,
              item: output,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            event("response.function_call_arguments.delta", {
              type: "response.function_call_arguments.delta",
              item_id: output.id,
              output_index: 0,
              delta: output.arguments,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            event("response.function_call_arguments.done", {
              type: "response.function_call_arguments.done",
              item_id: output.id,
              output_index: 0,
              arguments: output.arguments,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            event("response.output_item.done", {
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
            event("response.output_item.added", {
              type: "response.output_item.added",
              output_index: 0,
              item: { ...output, content: [] },
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            event("response.content_part.added", {
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
              event("response.output_text.delta", {
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
            event("response.output_text.done", {
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
            event("response.content_part.done", {
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
            event("response.output_item.done", {
              type: "response.output_item.done",
              output_index: 0,
              item: output,
            }),
          ),
        );
      }
      controller.enqueue(
        encoder.encode(
          event("response.completed", {
            type: "response.completed",
            response: result,
          }),
        ),
      );
      controller.close();
    },
  });
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
  const toolsRequested = hasTools(tools);
  if (toolsRequested)
    request.messages = injectToolPrompt(
      request.messages,
      tools,
      request.tool_choice,
    );

  const { endpoint, body } = buildUpstreamRequest(request, model);
  const upstream = await fetch(endpointUrl(endpoint, c.env.UPSTREAM_CHAT_URL), {
    method: "POST",
    headers: buildUpstreamHeaders(await c.get("sessionToken")(), c.env),
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(CHAT_TIMEOUT),
  });
  if (!upstream.ok || !upstream.body) {
    const detail = (await upstream.text().catch(() => "")).trim();
    throw upstreamError(
      upstream.status,
      detail
        ? `Upstream returned ${upstream.status}: ${detail.slice(0, 300)}`
        : `Upstream returned ${upstream.status} with no message.`,
    );
  }

  const parsed = await collectUpstream(upstream.body);
  const rawContent = parsed.content;
  const toolCall = toolsRequested
    ? tryParseRelayToolCall(rawContent, request.tools)
    : null;
  const split = splitReasoningContent(rawContent);
  const visibleContent = split.content || split.reasoningContent;
  const content = toolCall
    ? ""
    : parsed.citations.length
      ? inlineCitationLinks(visibleContent, parsed.citations) +
        formatCitations(parsed.citations)
      : visibleContent;
  const result = chatResultToResponses(
    model.id,
    content,
    toolCall,
    request.messages,
    undefined,
    split.reasoningContent,
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

import { Hono } from "hono";
import { CHAT_TIMEOUT } from "../constants/timeouts";
import type { Env, Variables } from "../types/env";
import type {
  ResponsesRequest,
  ResponsesResponse,
  ResponsesWebSearchCallOutput,
} from "../types/responses";
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
  splitReasoningContent,
} from "../utils/upstream-stream";
import {
  gateToolCalls,
  injectReasoningPrompt,
  injectToolPrompt,
  normalizeOpenAITools,
  stripToolCallMarkup,
  rejectedToolCallStub,
  toolCallRetrySignal,
  tryParseRelayToolCall,
  tryParseRelayToolCalls,
} from "../utils/tool-shim";
import {
  webSearch,
  webFetchFromSearchResults,
  type WebSearchResult,
} from "../utils/cloudflare-search-api";
import { declinedToolUse, needsCurrentWebSearch, sanitizeSearchQuery } from "../utils/search-query";
import { compactSearchToolResult } from "../utils/search-tool-context";
import { resolveCitationTitles } from "../utils/citation-titles";
import {
  resolveReasoningEffort,
} from "../utils/reasoning-prefill";

const MAX_HOSTED_TOOL_STEPS = 8;
const MAX_WEB_SEARCH_CALLS = 4;
const MAX_SEARCH_RESULTS_PER_CALL = 50;

const responses = new Hono<{ Bindings: Env; Variables: Variables }>();

function extractResponsesText(text: string): string {
  const dataLines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .filter((line) => line && line !== "[DONE]");
  if (dataLines.length) {
    const parts = dataLines
      .map((line) => extractResponsesJsonText(line))
      .filter((part): part is string => part !== null);
    if (parts.length) return parts.join("");
  }
  return extractResponsesJsonText(text) ?? text;
}

function extractResponsesJsonText(text: string): string | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{")) return null;
  try {
    const value = JSON.parse(trimmed) as Record<string, unknown>;
    if (typeof value.output_text === "string") return value.output_text;
    if (typeof value.delta === "string") return value.delta;
    if (Array.isArray(value.choices)) {
      const parts = value.choices.flatMap((choice) => {
        const message = (choice as Record<string, unknown>).message;
        const content =
          message && typeof message === "object"
            ? (message as Record<string, unknown>).content
            : undefined;
        return typeof content === "string" ? [content] : [];
      });
      if (parts.length) return parts.join("");
    }
    if (!Array.isArray(value.output)) return null;
    const parts: string[] = [];
    for (const item of value.output) {
      if (!item || typeof item !== "object") continue;
      const content = (item as Record<string, unknown>).content;
      if (!Array.isArray(content)) continue;
      for (const part of content) {
        if (!part || typeof part !== "object") continue;
        const partText = (part as Record<string, unknown>).text;
        if (typeof partText === "string") parts.push(partText);
      }
    }
    return parts.length ? parts.join("") : null;
  } catch {
    return null;
  }
}

export function addHostedWebSearches(
  result: ResponsesResponse,
  searches: readonly { query: string; results: WebSearchResult[] }[],
): ResponsesResponse {
  if (searches.length === 0) return result;
  const combined = new Map<string, WebSearchResult>();
  for (const search of searches) {
    for (const item of search.results) combined.set(item.url, item);
  }
  const enriched = addHostedWebSearch(
    result,
    searches[0]!.query,
    [...combined.values()],
  );
  const calls: ResponsesWebSearchCallOutput[] = searches.map((search) => ({
    type: "web_search_call",
    id: `ws_${crypto.randomUUID().replace(/-/g, "")}`,
    status: "completed",
    action: { type: "search", query: search.query },
    results: search.results,
  }));
  const output = enriched.output.flatMap((item, index) =>
    index === 0 && item.type === "web_search_call" ? calls : [item],
  );
  return { ...enriched, output };
}

function citationTitle(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

export async function addPerplexityCitations(
  result: ResponsesResponse,
  citations: readonly string[],
  options: { searchUrl?: string; searchToken?: string } = {},
): Promise<ResponsesResponse> {
  if (citations.length === 0) return result;
  const message = result.output.find((item) => item.type === "message");
  if (!message || !message.content[0]) return result;

  const titles = await resolveCitationTitles(citations, options);
  const textPart = message.content[0];
  const annotations: unknown[] = [...textPart.annotations];
  const marker = /\[(\d+)\]/g;
  let match: RegExpExecArray | null;
  let text = "";
  let cursor = 0;
  while ((match = marker.exec(textPart.text)) !== null) {
    const index = Number(match[1]);
    const url = citations[index - 1];
    if (!url) continue;
    text += textPart.text.slice(cursor, match.index);
    const markerStart = text.length;
    text += match[0];
    annotations.push({
      type: "url_citation",
      start_index: markerStart,
      end_index: markerStart + match[0].length,
      url,
      title: titles.get(url) ?? citationTitle(url),
    });
    cursor = match.index + match[0].length;
  }
  text += textPart.text.slice(cursor);

  const updatedMessage = {
    ...message,
    content: [{ ...textPart, text, annotations }],
  };
  return {
    ...result,
    output: result.output.map((item) =>
      item === message ? updatedMessage : item,
    ),
    output_text: text,
  };
}
export function addHostedWebSearch(
  result: ResponsesResponse,
  query: string,
  results: readonly WebSearchResult[],
): ResponsesResponse {
  const search: ResponsesWebSearchCallOutput = {
    type: "web_search_call",
    id: `ws_${crypto.randomUUID().replace(/-/g, "")}`,
    status: "completed",
    action: { type: "search", query },
    results: [...results],
  };
  const message = result.output.find((item) => item.type === "message");
  if (!message || results.length === 0) {
    return { ...result, output: [search, ...result.output] };
  }

  const textPart = message.content[0];
  if (!textPart) return { ...result, output: [search, ...result.output] };
  const annotations: unknown[] = [...textPart.annotations];
  const marker = /\[(\d+)\]/g;
  let match: RegExpExecArray | null;
  let text = "";
  let cursor = 0;
  while ((match = marker.exec(textPart.text)) !== null) {
    const index = Number(match[1]);
    const source = results[index - 1];
    if (!source) continue;
    text += textPart.text.slice(cursor, match.index);
    const markerStart = text.length;
    text += match[0];
    annotations.push({
      type: "url_citation",
      start_index: markerStart,
      end_index: markerStart + match[0].length,
      url: source.url,
      title: source.title,
    });
    cursor = match.index + match[0].length;
  }
  text += textPart.text.slice(cursor);

  if (annotations.length === textPart.annotations.length && results.length > 0) {
    for (const source of results) {
      annotations.push({
        type: "url_citation",
        start_index: 0,
        end_index: text.length,
        url: source.url,
        title: source.title,
      });
    }
  }

  const updatedMessage = {
    ...message,
    content: [
      {
        ...textPart,
        text,
        annotations,
      },
    ],
  };
  const output = result.output.map((item) =>
    item === message ? updatedMessage : item,
  );
  return {
    ...result,
    output: [search, ...output],
    output_text: text,
  };
}
function event(
  type: string,
  data: Record<string, unknown>,
  sequenceNumber: number,
): string {
  return `event: ${type}\ndata: ${JSON.stringify({
    ...data,
    sequence_number: sequenceNumber,
  })}\n\n`;
}

export function streamResponse(
  result: ResponsesResponse,
  toolCall: ReturnType<typeof tryParseRelayToolCall>,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      let sequenceNumber = 0;
      const send = (type: string, data: Record<string, unknown>) =>
        event(type, data, sequenceNumber++);
      const response = { ...result, status: "in_progress" };
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
      for (const [outputIndex, output] of result.output.entries()) {
        if (output.type !== "web_search_call") continue;
        const item = { ...output, status: "in_progress" };
        controller.enqueue(
          encoder.encode(
            send("response.output_item.added", {
              type: "response.output_item.added",
              output_index: outputIndex,
              item,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.web_search_call.in_progress", {
              type: "response.web_search_call.in_progress",
              output_index: outputIndex,
              item_id: output.id,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.web_search_call.searching", {
              type: "response.web_search_call.searching",
              output_index: outputIndex,
              item_id: output.id,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.web_search_call.completed", {
              type: "response.web_search_call.completed",
              output_index: outputIndex,
              item_id: output.id,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.output_item.done", {
              type: "response.output_item.done",
              output_index: outputIndex,
              item: output,
            }),
          ),
        );
      }

      const outputIndex = result.output.findIndex(
        (item) => item.type !== "web_search_call",
      );
      const output = result.output[outputIndex];
      if (toolCall && output?.type === "function_call") {
        controller.enqueue(
          encoder.encode(
            send("response.output_item.added", {
              type: "response.output_item.added",
              output_index: outputIndex,
              item: output,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.function_call_arguments.delta", {
              type: "response.function_call_arguments.delta",
              item_id: output.id,
              output_index: outputIndex,
              delta: output.arguments,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.function_call_arguments.done", {
              type: "response.function_call_arguments.done",
              item_id: output.id,
              output_index: outputIndex,
              arguments: output.arguments,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.output_item.done", {
              type: "response.output_item.done",
              output_index: outputIndex,
              item: output,
            }),
          ),
        );
      } else if (output?.type === "reasoning") {
        const text = output.content.map((part) => part.text).join("");
        controller.enqueue(
          encoder.encode(
            send("response.output_item.added", {
              type: "response.output_item.added",
              output_index: outputIndex,
              item: output,
            }),
          ),
        );
        if (text) {
          controller.enqueue(
            encoder.encode(
              send("response.reasoning_text.delta", {
                type: "response.reasoning_text.delta",
                item_id: output.id,
                output_index: outputIndex,
                content_index: 0,
                delta: text,
              }),
            ),
          );
        }
        controller.enqueue(
          encoder.encode(
            send("response.output_item.done", {
              type: "response.output_item.done",
              output_index: outputIndex,
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
              output_index: outputIndex,
              item: { ...output, content: [] },
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.content_part.added", {
              type: "response.content_part.added",
              item_id: output.id,
              output_index: outputIndex,
              content_index: 0,
              part: { ...part, text: "" },
            }),
          ),
        );
        if (text) {
          controller.enqueue(
            encoder.encode(
              send("response.output_text.delta", {
                type: "response.output_text.delta",
                item_id: output.id,
                output_index: outputIndex,
                content_index: 0,
                delta: text,
              }),
            ),
          );
        }
        for (const [annotationIndex, annotation] of part?.annotations.entries() ?? []) {
          controller.enqueue(
            encoder.encode(
              send("response.output_text.annotation.added", {
                type: "response.output_text.annotation.added",
                item_id: output.id,
                output_index: outputIndex,
                content_index: 0,
                annotation_index: annotationIndex,
                annotation,
              }),
            ),
          );
        }

        controller.enqueue(
          encoder.encode(
            send("response.output_text.done", {
              type: "response.output_text.done",
              item_id: output.id,
              output_index: outputIndex,
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
              output_index: outputIndex,
              content_index: 0,
              part,
            }),
          ),
        );
        controller.enqueue(
          encoder.encode(
            send("response.output_item.done", {
              type: "response.output_item.done",
              output_index: outputIndex,
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
  const reasoningRequested = Boolean(request.reasoning_effort);
  // Upstream is still chatplayground chat. Luna only emits reliable tool
  // calls / answers when reasoning_effort is "none" — same as /v1/chat.
  // Count tools after OpenAI web_search_preview → web_search/web_fetch mapping.
  const lunaWithTools =
    model.modelName.toLowerCase().includes("luna") &&
    (normalizeOpenAITools(request.tools).length > 0 ||
      normalizeOpenAITools(raw.tools).length > 0);
  const reasoningEffort = lunaWithTools
    ? "none"
    : resolveReasoningEffort(model.modelName, request.reasoning_effort);
  if (reasoningEffort) request.reasoning_effort = reasoningEffort;
  if (reasoningRequested && reasoningEffort && reasoningEffort !== "none") {
    request.messages = injectReasoningPrompt(request.messages, reasoningEffort);
  }
  if (request.messages.length === 0)
    throw invalidRequest("'input' must contain at least one message.", "input");
  const tools = request.tools ?? [];
  const webSearchRequested = tools.some(
    (tool) =>
      tool.function.name === "web_search" || tool.function.name === "web_fetch",
  );
  const toolsForModel = [...tools];
  if (
    webSearchRequested &&
    !toolsForModel.some((tool) => tool.function.name === "web_search")
  ) {
    toolsForModel.push({
      type: "function",
      function: {
        name: "web_search",
        description: "Search the web for current evidence using a focused query.",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string" },
            max_results: { type: "integer", minimum: 1, maximum: MAX_SEARCH_RESULTS_PER_CALL },
          },
          required: ["query"],
        },
      },
    });
  }
  if (
    webSearchRequested &&
    !toolsForModel.some((tool) => tool.function.name === "web_fetch")
  ) {
    toolsForModel.push({
      type: "function",
      function: {
        name: "web_fetch",
        description: "Read the result snippet for a URL returned by web_search.",
        parameters: {
          type: "object",
          properties: { url: { type: "string" } },
          required: ["url"],
        },
      },
    });
  }
  const searches: Array<{ query: string; results: WebSearchResult[] }> = [];
  const fetchedPages = new Map<string, WebSearchResult>();
  const toolsRequested = toolsForModel.length > 0;
  const latestUserText = [...request.messages].reverse().reduce((found, message) => {
    if (found || message.role !== "user") return found;
    return typeof message.content === "string" ? message.content : found;
  }, "");
  const forceCurrentSearch = webSearchRequested && needsCurrentWebSearch(latestUserText);
  const hasToolResult = request.messages.some((message) => message.role === "tool");
  if (toolsRequested && !hasToolResult && request.tool_choice !== "none") {
    request.tool_choice = request.tool_choice ?? "required";
  }
  if (toolsRequested) {
    // Search is model-directed: do not call the search service merely because
    // the request advertises a web-search tool. Execute it only after the
    // model emits a parsed web_search call below. Current-events prompts are
    // the exception: web_search is required before any answer.
    if (forceCurrentSearch && !hasToolResult) request.tool_choice = "required";
    else if (webSearchRequested) request.tool_choice = request.tool_choice ?? "auto";
    request.tools = toolsForModel;
    request.messages = injectToolPrompt(
      request.messages,
      toolsForModel,
      request.tool_choice,
      model.endpoint === "perplexity" ? "perplexity" : undefined,
    );
  }

  const sessionToken = await c.get("sessionToken")();
  let rawContent = "";
  let citations: Awaited<ReturnType<typeof collectUpstream>>["citations"] = [];
  let toolCall: ReturnType<typeof tryParseRelayToolCall> = null;
  let chatId: string | null = null;
  let upstream: Response | null = null;
  let parsedContent: Awaited<ReturnType<typeof collectUpstream>> | null = null;
  let toolGate: ReturnType<typeof gateToolCalls> = { calls: [], rejections: [] };

  for (let step = 0; step <= MAX_HOSTED_TOOL_STEPS; step++) {
    const built = buildUpstreamRequest(request, model);
    upstream = await fetch(endpointUrl(built.endpoint, c.env.UPSTREAM_CHAT_URL), {
      method: "POST",
      headers: buildUpstreamHeaders(sessionToken, c.env),
      body: JSON.stringify(built.body),
      signal: AbortSignal.timeout(CHAT_TIMEOUT),
    });
    if (!upstream.ok || !upstream.body) {
      const detail = (await upstream.text().catch(() => "")).trim();
      const message = detail
        ? `Upstream returned ${upstream.status}: ${detail.slice(0, 300)}`
        : `Upstream returned ${upstream.status} with no message.`;
      if (raw.stream) {
        return new Response(
          event("error", {
            type: "error",
            error: {
              code: `upstream_${upstream.status}`,
              type: upstream.status === 429 ? "rate_limit_error" : "upstream_error",
              message,
            },
          }, 0),
          {
            status: 200,
            headers: {
              "content-type": "text/event-stream; charset=utf-8",
              "cache-control": "no-cache, no-transform",
            },
          },
      );
      }
      throw upstreamError(upstream.status, message);
    }

    parsedContent = await collectUpstream(upstream.body);
    rawContent = extractResponsesText(parsedContent.content);
    toolGate = toolsRequested
      ? gateToolCalls(rawContent, request.tools)
      : { calls: [], rejections: [] };
    // Wrong/invalid tool calls: retry once with the gate rejection signal.
    // Skip the retry when a valid call already exists — avoids Agora timeout retries.
    if (toolsRequested && toolGate.rejections.length > 0 && toolGate.calls.length === 0) {
      request.messages = [
        ...request.messages,
        { role: "assistant", content: rejectedToolCallStub(toolGate.rejections) },
        { role: "user", content: toolCallRetrySignal(toolGate.rejections) },
      ];
      const gateBuilt = buildUpstreamRequest(request, model);
      const gateRetry = await fetch(endpointUrl(gateBuilt.endpoint, c.env.UPSTREAM_CHAT_URL), {
        method: "POST",
        headers: buildUpstreamHeaders(sessionToken, c.env),
        body: JSON.stringify(gateBuilt.body),
        signal: AbortSignal.timeout(CHAT_TIMEOUT),
      });
      if (gateRetry.ok && gateRetry.body) {
        const gateParsed = await collectUpstream(gateRetry.body);
        const gated = gateToolCalls(extractResponsesText(gateParsed.content), request.tools);
        if (gated.rejections.length === 0) {
          rawContent = extractResponsesText(gateParsed.content);
          parsedContent = gateParsed;
          toolGate = gated;
        } else {
          toolGate = { calls: [], rejections: gated.rejections };
          rawContent = stripToolCallMarkup(extractResponsesText(gateParsed.content)) || stripToolCallMarkup(rawContent);
          parsedContent = gateParsed;
        }
      } else {
        toolGate = { calls: [], rejections: toolGate.rejections };
        rawContent = stripToolCallMarkup(rawContent);
      }
    } else if (toolsRequested && /<TOOL_CALL\b/i.test(rawContent) && toolGate.calls.length === 0) {
      rawContent = stripToolCallMarkup(rawContent);
    }
    const toolCalls = toolGate.calls;
    toolCall = toolCalls[0] ?? null;
    if (!toolCall && !hasToolResult && searches.length === 0 && (forceCurrentSearch || declinedToolUse(rawContent))) {
      const query = sanitizeSearchQuery(latestUserText);
      if (query) {
        rawContent = `<TOOL_CALL>\ntool: web_search\nparams:\n${JSON.stringify({ query, max_results: 5 })}\n</TOOL_CALL>`;
        toolCall = tryParseRelayToolCall(rawContent, request.tools);
      }
    }
    // One call per step so multi-search turns don't look stuck.
    const calls = (toolCall && toolCalls.length === 0 ? [toolCall] : toolCalls).slice(0, 1);
    if (calls.length === 0 || !webSearchRequested || step === MAX_HOSTED_TOOL_STEPS) break;

    let delegated: typeof toolCall = null;
    for (const call of calls) {
    let toolResult: unknown;
    let toolName = call.function.name;
    let args: Record<string, unknown> = {};
    try {
      const parsedArgs: unknown = JSON.parse(call.function.arguments);
      if (parsedArgs && typeof parsedArgs === "object" && !Array.isArray(parsedArgs)) {
        args = parsedArgs as Record<string, unknown>;
      }
    } catch {
      toolResult = { error: "Tool arguments were not valid JSON." };
    }

    if (call.function.name === "web_search") {
      if (searches.length >= MAX_WEB_SEARCH_CALLS) {
        toolResult = { error: "Search call limit reached for this response." };
      } else if (typeof args.query !== "string" || !args.query.trim()) {
        toolResult = { error: "A non-empty query is required." };
      } else {
        const query = sanitizeSearchQuery(args.query);
        const requestedCount = Number(args.max_results);
        const count = Number.isFinite(requestedCount)
          ? Math.min(MAX_SEARCH_RESULTS_PER_CALL, Math.max(1, Math.floor(requestedCount)))
          : 5;
        const results = await webSearch(query, {
          url: c.env.CLOUDFLARE_SEARCH_URL,
          token: c.env.CLOUDFLARE_SEARCH_TOKEN,
          count,
        });
        if (results.length === 0) {
          // Cloudflare + Jina both failed: hand web_search to the client.
          delegated = call;
          break;
        }
        searches.push({ query, results });
        for (const result of results) fetchedPages.set(result.url, result);
        toolResult = { query, count, results };
      }
    } else if (call.function.name === "web_fetch") {
      const url = typeof args.url === "string" ? args.url : "";
      const extracted = await webFetchFromSearchResults(url, fetchedPages);
      toolResult = extracted ?? {
        error: "This search service only returns result snippets and cannot fetch page text. Use the returned snippets or search for a more specific source.",
      };
    } else {
      break;
    }

    request.messages.push({
      role: "assistant",
      content: null,
      tool_calls: [call],
    });
    request.messages.push({
      role: "tool",
      name: call.function.name,
      content: compactSearchToolResult(
        JSON.stringify(toolResult ?? { error: "Tool did not return a result." }),
        call.function.name,
      ),
    });
    }
    if (delegated) {
      toolCall = delegated;
      break;
    }
    toolCall = null;
  }

  if (!parsedContent) throw upstreamError(502, "No upstream model response was received.");
  citations = parsedContent.citations;
  chatId = parsedContent.chatId;

  // Luna sometimes returns an empty or plan-only turn after tool results.
  // Retry once asking for the answer; only then fail soft.
  if (
    !toolCall &&
    request.messages.some((message) => message.role === "tool") &&
    (!rawContent.trim() || declinedToolUse(rawContent))
  ) {
    request.messages = [
      ...request.messages,
      ...(rawContent.trim()
        ? [{ role: "assistant" as const, content: rawContent }]
        : []),
      {
        role: "user",
        content:
          "The tool result is already above. Answer the user's question from it now. Do not say you will check, search, or look it up.",
      },
    ];
    const retryBuilt = buildUpstreamRequest(request, model);
    const retry = await fetch(endpointUrl(retryBuilt.endpoint, c.env.UPSTREAM_CHAT_URL), {
      method: "POST",
      headers: buildUpstreamHeaders(sessionToken, c.env),
      body: JSON.stringify(retryBuilt.body),
      signal: AbortSignal.timeout(CHAT_TIMEOUT),
    });
    if (retry.ok && retry.body) {
      const retryParsed = await collectUpstream(retry.body);
      const retryText = extractResponsesText(retryParsed.content).trim();
      const retryCall = toolsRequested
        ? tryParseRelayToolCall(retryText, request.tools)
        : null;
      if (retryText && !retryCall && !declinedToolUse(retryText)) {
        rawContent = retryParsed.content;
        citations = retryParsed.citations;
        if (retryParsed.chatId) chatId = retryParsed.chatId;
        parsedContent = retryParsed;
      }
    }
  }

  // Never return raw tool XML as the assistant answer — Agora treats that as
  // a failed tool turn and shows "retry 1/5".
  if (!toolCall) rawContent = stripToolCallMarkup(rawContent);

  let split = splitReasoningContent(rawContent);
  let visibleContent = split.content || split.reasoningContent;
  if (!visibleContent.trim() && !toolCall) {
    if (toolsRequested && toolGate.rejections.length > 0) {
      // Invalid tool call after gate retry: soft-complete instead of empty 200.
      // Empty 200s make Agora retry the whole Responses request.
      rawContent =
        "The previous tool call was invalid and was not executed. Please retry with a complete tool call block and valid JSON params.";
      split = splitReasoningContent(rawContent);
      visibleContent = split.content || split.reasoningContent;
    } else if (request.messages.some((message) => message.role === "tool")) {
      rawContent =
        "I could not produce a final answer from the tool results. Please retry the request.";
      split = splitReasoningContent(rawContent);
      visibleContent = split.content || split.reasoningContent;
    } else {
      throw upstreamError(200, "Provider returned HTTP 200 but no text content.");
    }
  }
  // Final belt-and-suspenders: if somehow markup survived, strip before output.
  const content = toolCall ? "" : stripToolCallMarkup(visibleContent);
  const baseResult = chatResultToResponses(
    model.id,
    content,
    toolCall,
    request.messages,
    undefined,
    split.reasoningContent,
  );
  const resultWithCitations = citations.length
    ? await addPerplexityCitations(baseResult, citations, {
        searchUrl: c.env.CLOUDFLARE_SEARCH_URL,
        searchToken: c.env.CLOUDFLARE_SEARCH_TOKEN,
      })
    : baseResult;
  const result = searches.length
    ? addHostedWebSearches(resultWithCitations, searches)
    : resultWithCitations;
  console.log("RESPONSES RESULT", {
    model: model.id,
    stream: Boolean(raw.stream),
    internalTest: c.req.header("x-internal-test") === "combo-health-check",
    upstreamChars: rawContent.length,
    outputChars: result.output_text.length,
    outputItems: result.output.length,
    toolCall: Boolean(toolCall),
  });

  if (raw.stream) {
    return new Response(streamResponse(result, toolCall), {
      headers: {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
      },
    });
  }
  if (!toolCall) {
    return Response.json({
      ...result,
      // Compatibility with older OpenAI-shaped health checks, including
      // OmniRoute provider tests that inspect choices[].message.content.
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: result.output_text },
          finish_reason: "stop",
        },
      ],
    });
  }
  return Response.json(result);
});

export default responses;

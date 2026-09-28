import { Hono } from "hono";
import { CHAT_TIMEOUT } from "../constants/timeouts";
import type { Env, Variables } from "../types/env";
import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionUsage,
  OpenAIMessage,
} from "../types/openai";
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
  streamUpstreamAsOpenAI,
  streamUpstreamWithToolShim,
} from "../utils/upstream-stream";
import {
  injectReasoningPrompt,
  injectToolPrompt,
  normalizeOpenAITools,
  tryParseRelayToolCall,
} from "../utils/tool-shim";
import {
  webSearch,
  webFetchFromSearchResults,
  type WebSearchResult,
} from "../utils/cloudflare-search-api";
import { sanitizeSearchQuery } from "../utils/search-query";

const MAX_HOSTED_TOOL_STEPS = 8;
const MAX_WEB_SEARCH_CALLS = 4;
const MAX_SEARCH_RESULTS_PER_CALL = 50;

const chat = new Hono<{ Bindings: Env; Variables: Variables }>();

const CHAT_CACHE_TTL_SECONDS = 60 * 60 * 24 * 7; // 7 days

async function chatCacheKey(
  sessionToken: string,
  modelId: string,
  conversationId?: string,
): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(sessionToken),
  );
  const sessionHash = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  return `chat:${conversationId ?? `${sessionHash}:${modelId}`}`;
}

async function loadCachedChatId(env: Env, key: string): Promise<string | null> {
  if (!env.CHAT_CACHE) return null;
  return env.CHAT_CACHE.get(key);
}

async function saveCachedChatId(
  env: Env,
  key: string,
  chatId: string,
): Promise<void> {
  if (!env.CHAT_CACHE) return;
  await env.CHAT_CACHE.put(key, chatId, {
    expirationTtl: CHAT_CACHE_TTL_SECONDS,
  });
}

chat.post("/v1/chat/completions", async (c) => {
  const body = (await c.req
    .json()
    .catch(() => null)) as ChatCompletionRequest | null;

  if (!body || typeof body !== "object") {
    throw invalidRequest("Request body must be JSON.");
  }
  if (!body.model || typeof body.model !== "string") {
    throw invalidRequest("'model' is required.", "model");
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    throw invalidRequest("'messages' must be a non-empty array.", "messages");
  }

  const registry = await getModels(c.env);
  const model = findModel(body.model, registry);
  if (!model) throw modelNotFound(body.model);

  const sessionToken = await c.get("sessionToken")();

  // Reuse a prior upstream chat when the client hasn't explicitly set `user`
  // and there's a cached chatId for this conversation. This avoids spending
  // a brand-new chatplayground chat (and quota) on every single request.
  const conversationId = c.req.header("x-conversation-id") ?? undefined;
  const cacheKey = await chatCacheKey(sessionToken, model.id, conversationId);

  if (!body.user) {
    const cachedChatId = await loadCachedChatId(c.env, cacheKey);
    if (cachedChatId) {
      body.user = cachedChatId;
    }
  }

  // Preserve the caller's system prompt unless reasoning was explicitly
  // requested. A default reasoning instruction can consume the answer budget
  // and override application-level system instructions.
  const reasoningEffort = body.reasoning_effort;
  const originalMessages = body.messages;
  if (reasoningEffort) {
    body.messages = injectReasoningPrompt(body.messages, reasoningEffort);
  }

  // Keep both hosted search tools available to the model. The relay runs each
  // search/fetch call between model turns so any supported model can research
  // and then synthesize an answer without depending on client-side tool loops.
  const requestedTools = normalizeOpenAITools(body.tools);
  const webSearchRequested = requestedTools.some(
    (tool) => tool.function.name === "web_search" || tool.function.name === "web_fetch",
  );
  const toolsForModel = [...requestedTools];
  if (
    webSearchRequested &&
    !toolsForModel.some((tool) => tool.function.name === "web_search")
  ) {
    toolsForModel.push({
      type: "function",
      function: {
        name: "web_search",
        description: "Search the web for a focused query before opening result URLs.",
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
        description: "Read a URL returned by an earlier web_search call.",
        parameters: {
          type: "object",
          properties: { url: { type: "string" } },
          required: ["url"],
        },
      },
    });
  }
  const toolsRequested = toolsForModel.length > 0;
  if (webSearchRequested) {
    // Do not let a caller's `required` choice make the model repeat a tool call
    // after each hosted result instead of completing its response.
    body.tool_choice = "auto";
    body.messages.push({
      role: "system",
      content:
        "For web research, the first tool call must be web_search. Never call web_fetch before web_search has returned the URL. Use web_fetch on relevant search-result URLs when snippets are insufficient. After each tool result, decide whether a different focused web_search or another fetch is needed, then answer using the gathered evidence." ,
    });
  }
  body.tools = toolsRequested ? toolsForModel : undefined;
  if (toolsRequested) {
    body.messages = injectToolPrompt(
      body.messages,
      toolsForModel,
      body.tool_choice,
    );
  }

  const id = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);
  const searches: Array<{ query: string; results: WebSearchResult[] }> = [];
  const fetchedPages = new Map<string, WebSearchResult>();
  let rawContent = "";
  let citations: readonly string[] = [];
  let chatId: string | null = null;
  let toolCall: ReturnType<typeof tryParseRelayToolCall> = null;

  for (let step = 0; step <= MAX_HOSTED_TOOL_STEPS; step++) {
    const built = buildUpstreamRequest(body, model);
    console.log("UPSTREAM BODY:", JSON.stringify(built.body));
    const upstream = await fetch(
      endpointUrl(built.endpoint, c.env.UPSTREAM_CHAT_URL),
      {
        method: "POST",
        headers: buildUpstreamHeaders(sessionToken, c.env),
        body: JSON.stringify(built.body),
        signal: AbortSignal.timeout(CHAT_TIMEOUT),
      },
    );

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
    rawContent = parsed.content;
    citations = parsed.citations;
    chatId = parsed.chatId ?? chatId;
    if (chatId) {
      body.user = chatId;
      await saveCachedChatId(c.env, cacheKey, chatId);
    }
    toolCall = toolsRequested
      ? tryParseRelayToolCall(rawContent, body.tools)
      : null;

    if (
      !toolCall ||
      !webSearchRequested ||
      (toolCall.function.name !== "web_search" &&
        toolCall.function.name !== "web_fetch") ||
      step === MAX_HOSTED_TOOL_STEPS
    ) {
      break;
    }

    let args: Record<string, unknown> = {};
    try {
      const value: unknown = JSON.parse(toolCall.function.arguments);
      if (value && typeof value === "object" && !Array.isArray(value)) {
        args = value as Record<string, unknown>;
      }
    } catch {
      // A malformed call receives an explicit tool error, then the model may recover.
    }

    let toolResult: unknown;
    if (toolCall.function.name === "web_search") {
      if (searches.length >= MAX_WEB_SEARCH_CALLS) {
        toolResult = { error: "Search call limit reached for this response." };
      } else {
        const queryValue = args.query ?? args.search_query;
        if (typeof queryValue !== "string" || !queryValue.trim()) {
          toolResult = { error: "A non-empty search query is required." };
        } else {
          const query = sanitizeSearchQuery(queryValue);
          const requestedCount = Number(args.max_results ?? args.num_results);
          const count = Number.isFinite(requestedCount)
            ? Math.min(MAX_SEARCH_RESULTS_PER_CALL, Math.max(1, Math.floor(requestedCount)))
            : 5;
          const results = await webSearch(query, {
            url: c.env.CLOUDFLARE_SEARCH_URL,
            token: c.env.CLOUDFLARE_SEARCH_TOKEN,
            count,
          });
          searches.push({ query, results });
          for (const result of results) fetchedPages.set(result.url, result);
          toolResult = { query, count, results };
        }
      }
    } else {
      const url = typeof args.url === "string" ? args.url : "";
      const fetched = await webFetchFromSearchResults(url, fetchedPages);
      toolResult = fetched ?? {
        error: "That URL was not returned by web_search. Search first and fetch a URL from those results.",
      };
    }

    body.messages.push(
      { role: "assistant", content: null, tool_calls: [toolCall] },
      {
        role: "tool",
        name: toolCall.function.name,
        content: JSON.stringify(toolResult),
      },
    );
  }

  // Keep citations supplied by the upstream provider. Hosted search results
  // are evidence for the model, not an instruction to append a Sources list;
  // the model follows citation requirements from the caller's system prompt.
  const responseCitations = citations;
  const finalBody = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      const citationsTrailer = responseCitations.length
        ? `CITATIONS:${JSON.stringify(responseCitations)}`
        : "";
      const chatIdTrailer = chatId ? `CHAT_ID:${chatId}` : "";
      controller.enqueue(encoder.encode(`${rawContent}${citationsTrailer}${chatIdTrailer}`));
      controller.close();
    },
  });

  if (body.stream) {
    const onChatId = (value: string) => {
      void saveCachedChatId(c.env, cacheKey, value);
    };
    const sse = toolsRequested
      ? streamUpstreamWithToolShim(finalBody, {
          id,
          model: model.id,
          created,
          tools: body.tools,
          onChatId,
        })
      : streamUpstreamAsOpenAI(finalBody, {
          id,
          model: model.id,
          created,
          onChatId,
        });
    return new Response(sse, {
      status: 200,
      headers: {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
      },
    });
  }

  if (toolCall) {
    const toolResponse: ChatCompletionResponse = {
      id,
      object: "chat.completion",
      created,
      model: model.id,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: null, tool_calls: [toolCall] },
          finish_reason: "tool_calls",
        },
      ],
      usage: estimateUsage(originalMessages, rawContent),
    };
    return Response.json(toolResponse);
  }

  // Expose reasoning separately so Agora renders a collapsible thought block
  // instead of showing raw <think> tags in the assistant answer.
  const split = splitReasoningContent(rawContent);
  const answerText = split.content || split.reasoningContent;
  if (!answerText.trim() && !toolCall) {
    throw upstreamError(
      200,
      "Provider returned HTTP 200 but no text content.",
    );
  }
  const content =
    responseCitations.length === 0
      ? answerText
      : inlineCitationLinks(answerText, responseCitations) +
        formatCitations(responseCitations);

  const response: ChatCompletionResponse = {
    id,
    object: "chat.completion",
    created,
    model: model.id,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content,
          ...(split.reasoningContent
            ? { reasoning_content: split.reasoningContent }
            : {}),
        },
        finish_reason: "stop",
      },
    ],
    usage: estimateUsage(originalMessages, rawContent),
  };

  return Response.json(response);
});

// Crude usage estimate — chatplayground doesn't return token counts.
// ~4 chars per token. Multimodal content parts count text only.
function estimateUsage(
  messages: OpenAIMessage[],
  completion: string,
): ChatCompletionUsage {
  const promptChars = messages.reduce(
    (sum, m) => sum + textChars(m.content),
    0,
  );
  const prompt = Math.ceil(promptChars / 4);
  const comp = Math.ceil(completion.length / 4);
  return {
    prompt_tokens: prompt,
    completion_tokens: comp,
    total_tokens: prompt + comp,
  };
}

function textChars(content: OpenAIMessage["content"]): number {
  if (typeof content === "string") return content.length;
  if (!Array.isArray(content)) return 0;
  let n = 0;
  for (const part of content) {
    if (part.type === "text") n += part.text.length;
  }
  return n;
}

export default chat;

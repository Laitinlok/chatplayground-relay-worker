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
import { declinedToolUse, needsCurrentWebSearch, sanitizeSearchQuery } from "../utils/search-query";
import { resolveCitationTitles } from "../utils/citation-titles";
import {
  resolveReasoningEffort,
} from "../utils/reasoning-prefill";

const chat = new Hono<{ Bindings: Env; Variables: Variables }>();

const CHAT_CACHE_TTL_SECONDS = 60 * 60 * 24 * 7; // 7 days
const MAX_SEARCH_RESULTS_PER_CALL = 50;

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

  // Luna omits reasoning_effort from most clients. Default it to medium so
  // the tool-selection draft runs; an explicit "none" still disables it.
  const reasoningRequested = Boolean(body.reasoning_effort);
  // Upstream chatplayground chat: Luna only emits reliable function calls
  // when reasoning_effort is "none". Any other effort makes it ignore tools
  // or return empty/plan-only replies. /v1/responses uses the same rule.
  const lunaWithTools =
    model.modelName.toLowerCase().includes("luna") &&
    normalizeOpenAITools(body.tools).length > 0;
  const reasoningEffort = lunaWithTools
    ? "none"
    : resolveReasoningEffort(model.modelName, body.reasoning_effort);
  if (reasoningEffort) body.reasoning_effort = reasoningEffort;
  const originalMessages = body.messages;
  // Only an explicit effort rewrites the caller's system prompt. Luna's
  // medium default is for the hidden tool draft, not a visible reasoning mode.
  if (reasoningRequested && reasoningEffort) {
    body.messages = injectReasoningPrompt(body.messages, reasoningEffort);
  }

  // Advertise the relay's hosted search tools as ordinary OpenAI function tools.
  // Chat Completions returns calls to Agora/the client; unlike Responses, this
  // endpoint does not execute web_search or web_fetch inside the Worker.
  const requestedTools = normalizeOpenAITools(body.tools);
  const webSearchToolRequested = requestedTools.some(
    (tool) => tool.function.name === "web_search",
  );
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
  const latestUserText = (() => {
    const latestUser = [...originalMessages].reverse().find((message) => message.role === "user");
    return typeof latestUser?.content === "string"
      ? latestUser.content
      : Array.isArray(latestUser?.content)
        ? latestUser.content.filter((part) => part.type === "text").map((part) => part.text).join(" ")
        : "";
  })();
  // "latest/today/news" cannot be answered from memory. Force the search
  // call when the client already exposed web_search.
  const forceCurrentSearch =
    webSearchRequested && needsCurrentWebSearch(latestUserText);
  if (forceCurrentSearch && !originalMessages.some((message) => message.role === "tool")) body.tool_choice = "required";
  // First tool turn must call something. Later turns, which already carry a
  // tool result, go back to auto so the model can answer.
  const hasToolResult = originalMessages.some((message) => message.role === "tool");
  if (toolsRequested && !hasToolResult && body.tool_choice !== "none") {
    body.tool_choice = body.tool_choice ?? "required";
  }
  if (webSearchRequested) {
    // Keep search model-directed: the model must emit a web_search tool call,
    // which the relay executes and returns as a tool result. Do not pre-run
    // hosted search here, because that bypasses the tool-call exchange clients
    // such as Agora expect to see and misses the model's own focused query.
    if (!forceCurrentSearch) body.tool_choice = body.tool_choice ?? "auto";
    body.messages.push({
      role: "system",
      content:
        hasToolResult
          ? toolsForModel.some((tool) => !/^web_(search|fetch)$/i.test(tool.function.name))
            ? "A tool result is already above. Continue the user's task. Prefer the next needed edit/write/replace <TOOL_CALL> over a prose-only reply when the task is unfinished."
            : "A tool result is already above. Answer the user's question from it in natural language. Do not announce that you will check or search."
          : forceCurrentSearch
            ? "The user asked for current information (latest, today, or news). You must call web_search before answering. Do not answer from memory."
            : "The user requested web research. You must begin by calling web_search with a focused query; do not answer before the search tool returns. Then use web_fetch on relevant returned URLs when snippets are insufficient. If evidence is still missing, make another distinct web_search call. After research is sufficient, answer normally without another tool call.",
    });
  }
  body.tools = toolsRequested ? toolsForModel : undefined;
  if (toolsRequested) {
    body.messages = injectToolPrompt(
      body.messages,
      toolsForModel,
      body.tool_choice,
      model.endpoint === "perplexity" ? "perplexity" : undefined,
    );
  }

  const id = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);
  let rawContent = "";
  let citations: readonly string[] = [];
  let chatId: string | null = null;
  let toolCall: ReturnType<typeof tryParseRelayToolCall> = null;

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
  chatId = parsed.chatId;
  if (chatId) await saveCachedChatId(c.env, cacheKey, chatId);
  let toolGate = toolsRequested
    ? gateToolCalls(rawContent, body.tools)
    : { calls: [], rejections: [] as string[] };
  toolCall = toolGate.calls[0] ?? null;

  // Invalid / wrong tool calls: do not forward them. Retry once with the
  // rejection reasons so the model can emit a corrected <TOOL_CALL>.
  // If we already have at least one valid call, forward it immediately —
  // an extra upstream round-trip here is what makes Agora show "retry 1/5".
  if (toolsRequested && toolGate.rejections.length > 0 && toolGate.calls.length === 0) {
    body.messages = [
      ...body.messages,
      { role: "assistant", content: rejectedToolCallStub(toolGate.rejections) },
      { role: "user", content: toolCallRetrySignal(toolGate.rejections) },
    ];
    const gateBuilt = buildUpstreamRequest(body, model);
    const gateRetry = await fetch(endpointUrl(gateBuilt.endpoint, c.env.UPSTREAM_CHAT_URL), {
      method: "POST",
      headers: buildUpstreamHeaders(sessionToken, c.env),
      body: JSON.stringify(gateBuilt.body),
      signal: AbortSignal.timeout(CHAT_TIMEOUT),
    });
    if (gateRetry.ok && gateRetry.body) {
      const gateParsed = await collectUpstream(gateRetry.body);
      const gated = gateToolCalls(gateParsed.content, body.tools);
      // Accept only a clean retry. Rejected again → drop tool calls rather
      // than send a wrong one to the client.
      if (gated.rejections.length === 0) {
        rawContent = gateParsed.content;
        citations = gateParsed.citations;
        if (gateParsed.chatId) {
          chatId = gateParsed.chatId;
          await saveCachedChatId(c.env, cacheKey, chatId);
        }
        toolGate = gated;
        toolCall = gated.calls[0] ?? null;
      } else {
        toolCall = null;
        toolGate = { calls: [], rejections: gated.rejections };
        rawContent = stripToolCallMarkup(gateParsed.content) || stripToolCallMarkup(rawContent);
      }
    } else {
      toolCall = null;
      rawContent = stripToolCallMarkup(rawContent);
    }
  } else if (toolsRequested && /<TOOL_CALL\b/i.test(rawContent) && !toolCall) {
    // Rejected/incomplete markup with no parseable call: never show raw XML.
    rawContent = stripToolCallMarkup(rawContent);
  }

  // A tool result is already present and the model only announced that it
  // will check. Ask once more for the answer itself, and keep the plan only
  // if the follow-up is empty.
  if (hasToolResult && !toolCall && declinedToolUse(rawContent)) {
    body.messages = [
      ...body.messages,
      { role: "assistant", content: rawContent },
      {
        role: "user",
        content:
          "The tool result is already above. Answer the user's question from it now. Do not say you will check, search, or look it up.",
      },
    ];
    const retryBuilt = buildUpstreamRequest(body, model);
    const retry = await fetch(endpointUrl(retryBuilt.endpoint, c.env.UPSTREAM_CHAT_URL), {
      method: "POST",
      headers: buildUpstreamHeaders(sessionToken, c.env),
      body: JSON.stringify(retryBuilt.body),
      signal: AbortSignal.timeout(CHAT_TIMEOUT),
    });
    if (retry.ok && retry.body) {
      const retryParsed = await collectUpstream(retry.body);
      const retryText = retryParsed.content.trim();
      const retryCall = toolsRequested ? tryParseRelayToolCall(retryText, body.tools) : null;
      if (retryText && !retryCall && !declinedToolUse(retryText)) {
        rawContent = retryParsed.content;
        citations = retryParsed.citations;
        if (retryParsed.chatId) {
          chatId = retryParsed.chatId;
          await saveCachedChatId(c.env, cacheKey, chatId);
        }
      }
    }
  }

  // Some upstream model families return an empty completion instead of the
  // requested tool-call envelope. If the client explicitly requested
  // web_search, emit a valid delegated tool call using the latest user prompt
  // so Agora/OpenAI clients can still execute the search themselves.
  if (
    !toolCall &&
    webSearchToolRequested &&
    !hasToolResult &&
    (!rawContent.trim() || forceCurrentSearch || declinedToolUse(rawContent))
  ) {
    const query = sanitizeSearchQuery(latestUserText);
    if (query) {
      rawContent = `<TOOL_CALL>\ntool: web_search\nparams:\n${JSON.stringify({ query, max_results: 5 })}\n</TOOL_CALL>`;
      toolCall = tryParseRelayToolCall(rawContent, body.tools);
    }
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
          searchUrl: c.env.CLOUDFLARE_SEARCH_URL,
          searchToken: c.env.CLOUDFLARE_SEARCH_TOKEN,
        })
      : streamUpstreamAsOpenAI(finalBody, {
          id,
          model: model.id,
          created,
          onChatId,
          searchUrl: c.env.CLOUDFLARE_SEARCH_URL,
          searchToken: c.env.CLOUDFLARE_SEARCH_TOKEN,
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
          message: { role: "assistant", content: "", tool_calls: toolGate.calls.length ? toolGate.calls : tryParseRelayToolCalls(rawContent, body.tools) },
          finish_reason: "tool_calls",
        },
      ],
      usage: estimateUsage(originalMessages, rawContent),
    };
    return Response.json(toolResponse);
  }

  // Expose reasoning separately so Agora renders a collapsible thought block
  // instead of showing raw <think> tags in the assistant answer.
  if (!toolCall) rawContent = stripToolCallMarkup(rawContent);
  const split = splitReasoningContent(rawContent);
  let answerText = split.content || split.reasoningContent;
  if (!answerText.trim() && !toolCall) {
    // Rejected tool markup was stripped. Prefer a soft stop over raw XML or a hard empty-200.
    if (toolsRequested && toolGate.rejections.length > 0) {
      answerText =
        "The previous tool call was invalid and was not executed. Please retry with a complete tool call block and valid JSON params.";
    } else {
      throw upstreamError(
        200,
        "Provider returned HTTP 200 but no text content.",
      );
    }
  }
  const citationTitles = await resolveCitationTitles(responseCitations, {
    searchUrl: c.env.CLOUDFLARE_SEARCH_URL,
    searchToken: c.env.CLOUDFLARE_SEARCH_TOKEN,
  });
  const content =
    responseCitations.length === 0
      ? answerText
      : inlineCitationLinks(answerText, responseCitations) +
        formatCitations(responseCitations, citationTitles);

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

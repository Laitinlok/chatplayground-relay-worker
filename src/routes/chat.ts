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
  formatWebSearchContext,
} from "../utils/cloudflare-search-api";
import { sanitizeSearchQuery } from "../utils/search-query";

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

  // Prompt-injection tool-calling shim: inject exactly one authoritative
  // tool prompt, then parse the model's reply back into OpenAI tool_calls.
  const requestedTools = normalizeOpenAITools(body.tools);
  const webSearchRequested = requestedTools.some(
    (tool) => tool.function.name === "web_search",
  );
  const toolsForModel = requestedTools.filter(
    (tool) => tool.function.name !== "web_search",
  );
  const toolsRequested = toolsForModel.length > 0;
  if (webSearchRequested) {
    const latestUserMessage = [...originalMessages]
      .reverse()
      .find((message) => message.role === "user");
    const query = sanitizeSearchQuery(
      typeof latestUserMessage?.content === "string"
        ? latestUserMessage.content
        : "",
    );
    const results = await webSearch(query, {
      url: c.env.CLOUDFLARE_SEARCH_URL,
      token: c.env.CLOUDFLARE_SEARCH_TOKEN,
    });
    body.messages = [
      {
        role: "system",
        content: formatWebSearchContext(query, results),
      },
      ...body.messages,
    ];
  }
  if (toolsRequested) {
    body.tools = toolsForModel;
    body.messages = injectToolPrompt(
      body.messages,
      toolsForModel,
      body.tool_choice,
    );
  } else {
    body.tools = undefined;
  }

  const { endpoint, body: upstreamBody } = buildUpstreamRequest(body, model);

  console.log("UPSTREAM BODY:", JSON.stringify(upstreamBody));
  const upstream = await fetch(endpointUrl(endpoint, c.env.UPSTREAM_CHAT_URL), {
    method: "POST",
    headers: buildUpstreamHeaders(sessionToken, c.env),
    body: JSON.stringify(upstreamBody),
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

  const id = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);

  if (body.stream) {
    const onChatId = (chatId: string) => {
      void saveCachedChatId(c.env, cacheKey, chatId);
    };

    const sse = toolsRequested
      ? streamUpstreamWithToolShim(upstream.body, {
          id,
          model: model.id,
          created,
          tools: body.tools,
          onChatId,
        })
      : streamUpstreamAsOpenAI(upstream.body, {
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

  const {
    content: rawContent,
    citations,
    chatId,
  } = await collectUpstream(upstream.body);

  if (chatId) {
    await saveCachedChatId(c.env, cacheKey, chatId);
  }

  const toolCall = toolsRequested
    ? tryParseRelayToolCall(rawContent, body.tools)
    : null;

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
    citations.length === 0
      ? answerText
      : inlineCitationLinks(answerText, citations) + formatCitations(citations);

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

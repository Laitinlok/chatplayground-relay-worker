import { Hono } from "hono";
import { CHAT_TIMEOUT } from "../constants/timeouts";
import type { Env, Variables } from "../types/env";
import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionUsage,
  OpenAIMessage,
} from "../types/openai";
import {
  invalidRequest,
  modelNotFound,
  upstreamError,
} from "../utils/errors";
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
  streamUpstreamAsOpenAI,
  streamUpstreamWithToolShim,
} from "../utils/upstream-stream";
import {
  hasTools,
  injectToolPrompt,
  tryParseRelayToolCall,
} from "../utils/tool-shim";

const chat = new Hono<{ Bindings: Env; Variables: Variables }>();

const CHAT_CACHE_TTL_SECONDS = 60 * 60 * 24 * 7; // 7 days

function chatCacheKey(clerkUserId: string, modelId: string, conversationId?: string): string {
  return `chat:${conversationId ?? `${clerkUserId}:${modelId}`}`;
}

async function loadCachedChatId(env: Env, key: string): Promise<string | null> {
  if (!env.CHAT_CACHE) return null;
  return env.CHAT_CACHE.get(key);
}

async function saveCachedChatId(env: Env, key: string, chatId: string): Promise<void> {
  if (!env.CHAT_CACHE) return;
  await env.CHAT_CACHE.put(key, chatId, { expirationTtl: CHAT_CACHE_TTL_SECONDS });
}

chat.post("/v1/chat/completions", async (c) => {
  const body = (await c.req.json().catch(() => null)) as
    | ChatCompletionRequest
    | null;

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

  const clerkUserId = c.get("clerkUserId");

  // Reuse a prior upstream chat when the client hasn't explicitly set `user`
  // and there's a cached chatId for this conversation. This avoids spending
  // a brand-new chatplayground chat (and quota) on every single request.
  const conversationId = c.req.header("x-conversation-id") ?? undefined;
  const cacheKey = chatCacheKey(clerkUserId, model.id, conversationId);

  if (!body.user) {
    const cachedChatId = await loadCachedChatId(c.env, cacheKey);
    if (cachedChatId) {
      body.user = cachedChatId;
    }
  }

// Prompt-injection tool-calling shim: inject exactly one authoritative
// tool prompt, then parse the model's reply back into OpenAI tool_calls.
  const toolsRequested = hasTools(body.tools);
  if (toolsRequested) {
    body.messages = injectToolPrompt(body.messages, body.tools!, body.tool_choice);
  }

  const { endpoint, body: upstreamBody } = buildUpstreamRequest(body, model);

  console.log("UPSTREAM BODY:", JSON.stringify(upstreamBody));
  const upstream = await fetch(endpointUrl(endpoint, c.env.UPSTREAM_CHAT_URL), {
    method: "POST",
    headers: buildUpstreamHeaders(clerkUserId, c.env),
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(CHAT_TIMEOUT),
  });

  if (!upstream.ok || !upstream.body) {
    throw upstreamError(
      upstream.status,
      `Upstream returned ${upstream.status}. Most likely cause: invalid X-Clerk-User-Id, unsupported model, or upstream outage.`,
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

  const { content: rawContent, citations, chatId } = await collectUpstream(upstream.body);

  if (chatId) {
    await saveCachedChatId(c.env, cacheKey, chatId);
  }

  const toolCall = toolsRequested ? tryParseRelayToolCall(rawContent, body.tools) : null;

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
      usage: estimateUsage(body.messages, rawContent),
    };
    return Response.json(toolResponse);
  }

  // Non-streaming path: inline-rewrite [N] citation markers as Markdown links
  // and append a sources block. Usage stays based on raw model output so the
  // relay-added link formatting doesn't inflate completion_tokens.
  const content =
    citations.length === 0
      ? rawContent
      : inlineCitationLinks(rawContent, citations) +
        formatCitations(citations);

  const response: ChatCompletionResponse = {
    id,
    object: "chat.completion",
    created,
    model: model.id,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: estimateUsage(body.messages, rawContent),
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

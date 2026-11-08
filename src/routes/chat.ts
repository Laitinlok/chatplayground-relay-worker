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
  formatToolsText,
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
  missingToolCallRetrySignal,
  looksLikeNonXmlToolAttempt,
  tryParseRelayToolCall,
} from "../utils/tool-shim";
import { declinedToolUse, needsCurrentWebSearch } from "../utils/search-query";
import { bridgeInlineImages, uploadTextAttachment } from "../utils/image-bridge";
import { formatHistoryText } from "../utils/upstream-request";
import {
  isFetchTool,
  isSearchTool,
  isVeryBasicQuery,
  prioritizeSearchTools,
  createSearchToolCall,
  selectToolForAutoChoice,
} from "../utils/tool-selector";
import { resolveCitationTitles } from "../utils/citation-titles";
import { resolveReasoningEffort } from "../utils/reasoning-prefill";
import { CHAT_CACHE_TTL_SECONDS, chatCacheKey } from "../utils/chat-cache";

const chat = new Hono<{ Bindings: Env; Variables: Variables }>();

function latestUserText(messages: OpenAIMessage[]): string {
  const latestUser = [...messages]
    .toReversed()
    .find((message) => message.role === "user");
  return typeof latestUser?.content === "string"
    ? latestUser.content
    : Array.isArray(latestUser?.content)
      ? latestUser.content
          .filter((part) => part.type === "text")
          .map((part) => part.text)
          .join(" ")
      : "";
}

function logRequestShape(body: ChatCompletionRequest): void {
  const messages = body.messages.map((message) => ({
    role: message.role,
    name: message.name,
    tool_call_id: message.tool_call_id,
    content:
      typeof message.content === "string"
        ? message.content.slice(0, 500)
        : undefined,
    tool_calls: message.tool_calls?.map((call) => ({
      id: call.id,
      name: call.function.name,
      arguments: call.function.arguments.slice(0, 1_000),
    })),
  }));
  console.log(
    JSON.stringify({
      event: "relay.chat.request",
      model: body.model,
      stream: body.stream === true,
      tool_choice: body.tool_choice ?? null,
      tools: normalizeOpenAITools(body.tools).map((tool) => ({
        name: tool.function.name,
        parameters: tool.function.parameters,
      })),
      messages,
    }),
  );
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

  if (c.env.DEBUG_REQUEST_LOGS === "true") logRequestShape(body);

  const registry = await getModels(c.env);
  const model = findModel(body.model, registry);
  if (!model) throw modelNotFound(body.model);
  const isLuna = model.modelName.toLowerCase().includes("luna");
  const promptOnly = /(?:grok|glm)/i.test(model.modelName);
  const sessionToken = await c.get("sessionToken")();
  await bridgeInlineImages(body.messages, c.env, sessionToken);
  const conversationId = c.req.header("x-conversation-id");
  const cacheKey = await chatCacheKey(
    sessionToken,
    model.id,
    c.env.CF_VERSION_METADATA?.id ?? "local",
    conversationId ?? body.user,
  );
  const hasHistory = body.messages.length > 1;
  const cachedChatId =
    hasHistory && c.env.CHAT_CACHE
      ? await c.env.CHAT_CACHE.get(cacheKey)
      : null;

  // Luna omits reasoning_effort from most clients. Default it to medium so
  // the tool-selection draft runs; an explicit "none" still disables it.
  const reasoningRequested = Boolean(body.reasoning_effort);
  // Upstream chatplayground chat: Luna only emits reliable function calls
  // when reasoning_effort is "none". Any other effort makes it ignore tools
  // or return empty/plan-only replies. /v1/responses uses the same rule.
  const lunaWithTools = isLuna && normalizeOpenAITools(body.tools).length > 0;
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

  const requestedTools = normalizeOpenAITools(body.tools);
  const latestUserIndex = originalMessages.findLastIndex(
    (message) => message.role === "user",
  );
  const currentTurnMessages = originalMessages.slice(latestUserIndex + 1);
  const hasToolResult = currentTurnMessages.some(
    (message) => message.role === "tool",
  );
  const hasRelaySearchResult = currentTurnMessages.some((message) => {
    if (message.role !== "tool") return false;
    if (
      typeof message.name === "string" &&
      requestedTools.some(
        (tool) => tool.function.name === message.name && isSearchTool(tool),
      )
    )
      return true;
    if (!message.tool_call_id) return false;
    return originalMessages.some(
      (item) =>
        item.role === "assistant" &&
        item.tool_calls?.some(
          (call) =>
            call.id === message.tool_call_id &&
            requestedTools.some(
              (tool) =>
                tool.function.name === call.function.name && isSearchTool(tool),
            ),
        ),
    );
  });
  const userText = latestUserText(body.messages);
  if (
    !promptOnly &&
    isVeryBasicQuery(userText) &&
    (body.tool_choice === undefined || body.tool_choice === "auto")
  ) {
    body.tool_choice = "none";
  }
  const toolsForModel = prioritizeSearchTools(
    requestedTools,
    hasRelaySearchResult || typeof body.tool_choice === "object",
    userText,
  );
  const toolsRequested =
    toolsForModel.length > 0 && body.tool_choice !== "none";
  const forcedToolChoice = body.tool_choice;
  if (
    typeof forcedToolChoice === "object" &&
    "function" in forcedToolChoice &&
    !toolsForModel.some(
      (tool) => tool.function.name === forcedToolChoice.function.name,
    )
  ) {
    // An initial forced fetch/scrape call refers to a tool hidden by the
    // search-first gate. Let the selector route this turn to discovery.
    body.tool_choice = "auto";
  }
  const catalogIsWebOnly = toolsForModel.every(
    (tool) => isSearchTool(tool) || isFetchTool(tool),
  );
  let selectedTool: ReturnType<typeof tryParseRelayToolCall> = null;
  const hasPriorUserTurn = originalMessages
    .slice(0, latestUserIndex)
    .some((message) => message.role === "user");
  if (toolsRequested) {
    body.tools = toolsForModel;
    selectedTool =
      !hasRelaySearchResult &&
      (promptOnly
        ? needsCurrentWebSearch(userText)
        : isSearchTool(toolsForModel[0]!) && needsCurrentWebSearch(userText))
        ? createSearchToolCall(userText, toolsForModel)
        : !hasPriorUserTurn || isVeryBasicQuery(userText) || !catalogIsWebOnly
          ? null
          : await selectToolForAutoChoice(
              body,
              model,
              c.env,
              sessionToken,
              hasRelaySearchResult,
            );
    // Pin a single tool only for search/fetch catalogs. Mixed catalogs
    // (documents, edits, terminal) stay on auto so the model can choose.
    if (selectedTool && catalogIsWebOnly) {
      body.tool_choice = {
        type: "function",
        function: { name: selectedTool.function.name },
      };
    } else if (selectedTool) {
      body.tool_choice = body.tool_choice ?? "auto";
    }
    body.messages = injectToolPrompt(
      body.messages,
      toolsForModel,
      body.tool_choice,
      model.endpoint === "perplexity" ? "perplexity" : undefined,
      false,
    );
  }

  const id = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);
  if (selectedTool && catalogIsWebOnly) {
    if (!body.stream) {
      return Response.json({
        id,
        object: "chat.completion",
        created,
        model: model.id,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              tool_calls: [selectedTool],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: estimateUsage(originalMessages, ""),
      } satisfies ChatCompletionResponse);
    }
    const source = new ReadableStream<Uint8Array>({
      start(controller) {
                controller.enqueue(
          new TextEncoder().encode(
            `<TOOL_CALL>\ntool: ${selectedTool.function.name}\nparams:\n${selectedTool.function.arguments}\n</TOOL_CALL>`,
          ),
        );
        controller.close();
      },
    });
    return new Response(
      streamUpstreamWithToolShim(source, {
        id,
        model: model.id,
        created,
        tools: body.tools,
      }),
      {
        headers: {
          "content-type": "text/event-stream; charset=utf-8",
          "cache-control": "no-cache, no-transform",
        },
      },
    );
  }

  const maxUserFiles = toolsRequested || hasPriorUserTurn ? 4 : 6;
  const rawFileUrls = body.file_ids ?? body.fileUrl?.split("|||") ?? [];
  if (!Array.isArray(rawFileUrls)) {
    throw invalidRequest("file_ids must be an array of HTTPS file URLs.", "file_ids");
  }
  const userFileUrls = rawFileUrls;
  if (
    userFileUrls.length > maxUserFiles ||
    userFileUrls.some((url) => {
      if (typeof url !== "string") return true;
      try {
        return new URL(url).protocol !== "https:";
      } catch {
        return true;
      }
    })
  ) {
    throw invalidRequest(
      `file_ids must contain at most ${maxUserFiles} HTTPS file URLs.`,
      "file_ids",
    );
  }
  let preparatoryChatId: string | undefined;
  if (toolsRequested && !hasPriorUserTurn && !promptOnly) {
    const callerSystem = originalMessages
      .filter((message) => message.role === "system" || message.role === "developer")
      .map((message) => typeof message.content === "string" ? message.content : "")
      .filter(Boolean)
      .join("\n\n");
    const setupMessages = injectToolPrompt(
      callerSystem ? [{ role: "system", content: callerSystem }] : [],
      toolsForModel,
      body.tool_choice,
      model.endpoint === "perplexity" ? "perplexity" : undefined,
      true,
    );
    const setup = buildUpstreamRequest(
      { ...body, messages: setupMessages },
      model,
    );
    const setupResponse = await fetch(
      endpointUrl(setup.endpoint, c.env.UPSTREAM_CHAT_URL),
      {
        method: "POST",
        headers: buildUpstreamHeaders(sessionToken, c.env),
        body: JSON.stringify(setup.body),
        signal: AbortSignal.timeout(CHAT_TIMEOUT),
      },
    );
    if (!setupResponse.ok || !setupResponse.body) {
      const detail = (await setupResponse.text().catch(() => "")).trim();
      throw upstreamError(
        setupResponse.status,
        detail ? `Tool setup failed: ${detail.slice(0, 200)}` : "Tool setup turn failed.",
      );
    }
    // Discard the setup acknowledgement; only its upstream chat ID is reused.
    preparatoryChatId =
      (await collectUpstream(setupResponse.body)).chatId ?? undefined;
  }

  const contextFileUrls = toolsRequested && !promptOnly
    ? await Promise.all([
        uploadTextAttachment(
          formatToolsText(toolsForModel),
          "tools.txt",
          c.env,
          sessionToken,
        ),
        uploadTextAttachment(
          formatHistoryText(originalMessages),
          "history.txt",
          c.env,
          sessionToken,
        ),
      ])
    : [];
  const fileUrls = [...userFileUrls, ...contextFileUrls];

  let rawContent = "";
  let citations: readonly string[] = [];
  let chatId: string | null = null;
  let toolCall: ReturnType<typeof tryParseRelayToolCall> = null;

  const built = buildUpstreamRequest(body, model, {
    chatId: promptOnly ? undefined : preparatoryChatId ?? cachedChatId ?? undefined,
    fileUrls,
    contextFilesOnly: toolsRequested && !promptOnly,
    promptOnly,
    preserveRoles: model.endpoint === "perplexity" || model.endpoint === "lmsys",
  });
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
  if (chatId) {
    await c.env.CHAT_CACHE?.put(cacheKey, chatId, {
      expirationTtl: CHAT_CACHE_TTL_SECONDS,
    });
  }
  let toolGate = toolsRequested
    ? gateToolCalls(rawContent, body.tools)
    : { calls: [], rejections: [] as string[] };
  toolCall = toolGate.calls[0] ?? null;

  // Invalid / wrong tool calls: do not forward them. Retry once with the
  // rejection reasons so the model can emit a corrected <TOOL_CALL>.
  // If we already have at least one valid call, forward it immediately —
  // an extra upstream round-trip here is what makes Agora show "retry 1/5".
  const toolChoiceRequiresCall =
    body.tool_choice === "required" ||
    (typeof body.tool_choice === "object" && "function" in body.tool_choice);
  const shouldRetryMissingTool =
    toolsRequested &&
    toolGate.calls.length === 0 &&
    (toolGate.rejections.length > 0 ||
      toolChoiceRequiresCall ||
      (hasToolResult && looksLikeNonXmlToolAttempt(rawContent)) ||
      (!hasToolResult &&
        (needsCurrentWebSearch(userText) || !catalogIsWebOnly) &&
        (looksLikeNonXmlToolAttempt(rawContent) ||
          /\b(?:not available|cannot|can['’]t|unable to|don['’]t have access)\b/i.test(rawContent))));
  if (shouldRetryMissingTool) {
    const retryUser = [
      userText,
      toolGate.rejections.length > 0
        ? toolCallRetrySignal(toolGate.rejections)
        : missingToolCallRetrySignal(body.tools ?? []),
    ]
      .filter(Boolean)
      .join("\n\n");
    body.messages = [
      ...body.messages,
      {
        role: "assistant",
        content:
          toolGate.rejections.length > 0
            ? rejectedToolCallStub(toolGate.rejections)
            : stripToolCallMarkup(rawContent) || "Skipped required tool call.",
      },
      { role: "user", content: retryUser },
    ];
    const gateBuilt = buildUpstreamRequest(body, model, {
      chatId: promptOnly ? undefined : chatId ?? cachedChatId ?? undefined,
      fileUrls,
      contextFilesOnly: toolsRequested && !promptOnly,
      promptOnly,
      preserveRoles: model.endpoint === "perplexity" || model.endpoint === "lmsys",
    });
    const gateRetry = await fetch(
      endpointUrl(gateBuilt.endpoint, c.env.UPSTREAM_CHAT_URL),
      {
        method: "POST",
        headers: buildUpstreamHeaders(sessionToken, c.env),
        body: JSON.stringify(gateBuilt.body),
        signal: AbortSignal.timeout(CHAT_TIMEOUT),
      },
    );
    if (gateRetry.ok && gateRetry.body) {
      const gateParsed = await collectUpstream(gateRetry.body);
      const gated = gateToolCalls(gateParsed.content, body.tools);
      if (gateParsed.chatId) {
        chatId = gateParsed.chatId;
        await c.env.CHAT_CACHE?.put(cacheKey, chatId, {
          expirationTtl: CHAT_CACHE_TTL_SECONDS,
        });
      }
      // Accept only a clean retry. Rejected again → drop tool calls rather
      // than send a wrong one to the client.
      if (gated.rejections.length === 0) {
        rawContent = gateParsed.content;
        citations = gateParsed.citations;
        chatId = gateParsed.chatId ?? chatId;
        if (gateParsed.chatId) {
          await c.env.CHAT_CACHE?.put(cacheKey, chatId, {
            expirationTtl: CHAT_CACHE_TTL_SECONDS,
          });
        }
        toolGate = gated;
        toolCall = gated.calls[0] ?? null;
      } else {
        toolCall = null;
        toolGate = { calls: [], rejections: gated.rejections };
        rawContent =
          stripToolCallMarkup(gateParsed.content) ||
          stripToolCallMarkup(rawContent);
      }
    } else {
      toolCall = null;
      rawContent = stripToolCallMarkup(rawContent);
    }
  } else if (toolsRequested && /<TOOL_CALL\b/i.test(rawContent) && !toolCall) {
    // Rejected/incomplete markup with no parseable call: never show raw XML.
    rawContent = stripToolCallMarkup(rawContent);
  }

  // their first completed upstream response is returned without another
  // potentially expensive provider round-trip.
  if (isLuna && hasToolResult && !toolCall && declinedToolUse(rawContent)) {
    body.messages = [
      ...body.messages,
      { role: "assistant", content: rawContent },
      {
        role: "user",
        content:
          "The tool result is already above. Answer the user's question from it now. Do not say you will check, search, or look it up.",
      },
    ];
    const retryBuilt = buildUpstreamRequest(body, model, {
      chatId: promptOnly ? undefined : chatId ?? cachedChatId ?? undefined,
      fileUrls,
      contextFilesOnly: toolsRequested && !promptOnly,
      promptOnly,
      preserveRoles: model.endpoint === "perplexity" || model.endpoint === "lmsys",
    });
    const retry = await fetch(
      endpointUrl(retryBuilt.endpoint, c.env.UPSTREAM_CHAT_URL),
      {
        method: "POST",
        headers: buildUpstreamHeaders(sessionToken, c.env),
        body: JSON.stringify(retryBuilt.body),
        signal: AbortSignal.timeout(CHAT_TIMEOUT),
      },
    );
    if (retry.ok && retry.body) {
      const retryParsed = await collectUpstream(retry.body);
      const retryText = retryParsed.content.trim();
      const retryCall = toolsRequested
        ? tryParseRelayToolCall(retryText, body.tools)
        : null;
      if (retryText && !retryCall && !declinedToolUse(retryText)) {
        rawContent = retryParsed.content;
        citations = retryParsed.citations;
        chatId = retryParsed.chatId ?? chatId;
        if (retryParsed.chatId) {
          await c.env.CHAT_CACHE?.put(cacheKey, retryParsed.chatId, {
            expirationTtl: CHAT_CACHE_TTL_SECONDS,
          });
        }
      }
    }
  }

  if (body.stream && toolCall && toolGate.calls.length) {
    rawContent = toolGate.calls
      .map(
        (call) =>
          `<TOOL_CALL>\ntool: ${call.function.name}\nparams:\n${call.function.arguments}\n</TOOL_CALL>`,
      )
      .join("\n");
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
      controller.enqueue(
        encoder.encode(`${rawContent}${citationsTrailer}`),
      );
      controller.close();
    },
  });

  if (body.stream) {
    const sse = toolsRequested
      ? streamUpstreamWithToolShim(finalBody, {
          id,
          model: model.id,
          created,
          tools: body.tools,
          searchUrl: c.env.CLOUDFLARE_SEARCH_URL,
          searchToken: c.env.CLOUDFLARE_SEARCH_TOKEN,
        })
      : streamUpstreamAsOpenAI(finalBody, {
          id,
          model: model.id,
          created,
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

  if (toolGate.calls.length > 0) {
    const toolResponse: ChatCompletionResponse = {
      id,
      object: "chat.completion",
      created,
      model: model.id,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            // Tool-only assistant turn. `null` avoids clients rendering this
            // as a blank final response before processing tool_calls.
            content: null,
            tool_calls: toolGate.calls,
          },
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
  if (content === null) return 0;
  if (typeof content === "string") return content.length;
  if (!Array.isArray(content)) return 0;
  let n = 0;
  for (const part of content) {
    if (part.type === "text") n += part.text.length;
  }
  return n;
}

export default chat;

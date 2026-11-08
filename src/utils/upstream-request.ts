import type { UpstreamEndpoint } from "../constants/endpoints";
import type { ModelEntry } from "../constants/models";
import type { ChatCompletionRequest } from "../types/openai";
import type {
  UpstreamChatRequest,
  UpstreamContentPart,
  UpstreamContentPartText,
  UpstreamMessage,
} from "../types/upstream";
import { compactSearchToolResult } from "./search-tool-context";
import { buildToolSystemPrompt } from "./tool-shim";
import type { OpenAITool } from "./tool-shim";

// Leave headroom for upstream-side normalization and Unicode length counting.
const MAX_UPSTREAM_MESSAGE_CHARS = 12_000;
const OMITTED_CONTEXT = "…(earlier context omitted)\n";

function flattenContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p): p is { type: string; text?: string } => p?.type === "text")
      .map((p) => p.text ?? "")
      .join("\n");
  }
  return "";
}

export function formatHistoryText(messages: ChatCompletionRequest["messages"]): string {
  return messages
    .filter(
      (message) =>
        message.role === "user" ||
        message.role === "assistant" ||
        message.role === "tool",
    )
    .map((message) => {
      const content = flattenContent(message.content).trim();
      const calls = message.tool_calls?.map((call) =>
        `Tool call ${call.function.name}: ${call.function.arguments}`,
      );
      const label =
        message.role === "user"
          ? "USER"
          : message.role === "tool"
            ? "TOOL_RESULT"
            : "RESPONSE";
      return [
        `<${label}>`,
        content,
        ...(calls ?? []),
        `</${label}>`,
      ].join("\n");
    })
    .join("\n\n");
}

export function formatToolsText(tools: OpenAITool[]): string {
  return JSON.stringify(
    tools.map((tool) => ({
      name: tool.function.name,
      description: tool.function.description ?? "",
      parameters: tool.function.parameters ?? {},
    })),
    null,
    2,
  );
}

export interface BuiltUpstreamRequest {
  endpoint: UpstreamEndpoint;
  body: UpstreamChatRequest;
}

export function buildUpstreamRequest(
  req: ChatCompletionRequest,
  model: ModelEntry,
  options: {
    chatId?: string;
    fileUrls?: string[];
    currentTurnOnly?: boolean;
    contextFilesOnly?: boolean;
    promptOnly?: boolean;
    preserveRoles?: boolean;
  } = {},
): BuiltUpstreamRequest {
  // Preserve multimodal parts when possible and normalize roles unsupported by
  // chatplayground before folding the transcript into a single user message.
  // chatplayground also has no concept of `tool_calls`: an assistant
  // message that only carries tool_calls has `content: null`, which the
  // upstream Azure/perplexity/lmsys endpoints reject outright ("400 Invalid
  // value for 'content': expected a string, got null"). Fold tool_calls and
  // tool-role results into plain text instead of dropping/nulling them.
  const lastAssistantTurn = req.messages.findLastIndex(
    (message) => message.role === "assistant",
  );
  const lastToolResult = req.messages.findLastIndex(
    (message) => message.role === "tool",
  );
  const lastUserTurn = req.messages.findLastIndex(
    (message) => message.role === "user",
  );
  const promptStart =
    lastToolResult >= 0
      ? req.messages.findLastIndex(
          (message, index) => index < lastToolResult && message.role === "user",
        )
      : lastUserTurn;
  const currentTurnStart = promptStart >= 0 ? promptStart : lastUserTurn;
  const sourceMessages = options.promptOnly
    ? [
        ...(req.tools?.length
          ? [
              {
                role: "user" as const,
                content: [
                  ...req.messages
                    .filter(
                      (message) =>
                        (message.role === "system" ||
                          message.role === "developer") &&
                        !flattenContent(message.content).includes(
                          "[relay-tool-prompt-v1]",
                        ),
                    )
                    .map((message) => flattenContent(message.content).trim())
                    .filter(Boolean),
                  buildToolSystemPrompt(
                    req.tools,
                    req.tool_choice,
                    undefined,
                    false,
                  ),
                  "Read history.txt as conversation history. Preserve caller system instructions. Use new search results for current facts.",
                ].join("\n\n"),
              },
            ]
          : []),
        ...req.messages.slice(Math.max(0, currentTurnStart)),
      ]
    : options.contextFilesOnly
      ? [
          {
            role: "user" as const,
            content: [
              ...req.messages
                .filter(
                  (message) =>
                    (message.role === "system" || message.role === "developer") &&
                    !flattenContent(message.content).includes(
                      "[relay-tool-prompt-v1]",
                    ),
                )
                .map((message) => flattenContent(message.content).trim())
                .filter(Boolean),
              ...req.messages
                .filter(
                  (message) =>
                    message.role === "system" &&
                    flattenContent(message.content).includes(
                      "[relay-tool-prompt-v1]",
                    ),
                )
                .map((message) => flattenContent(message.content).trim()),
              "Read tools.txt only for tool names and schemas; read history.txt only as conversation context. Follow caller system instructions.",
            ].join("\n\n"),
          },
          ...req.messages.slice(Math.max(0, currentTurnStart)),
        ]
      : options.currentTurnOnly && lastAssistantTurn >= 0
        ? [
            ...req.messages.filter(
              (message) =>
                message.role === "system" || message.role === "developer",
            ),
            ...req.messages.slice(Math.max(0, lastUserTurn)),
          ]
        : req.messages;
  const messages: UpstreamMessage[] = sourceMessages.flatMap((msg) => {
    const role =
      msg.role === "tool"
        ? "user"
        : msg.role === "developer"
          ? "system"
          : msg.role;

    let content: UpstreamMessage["content"];
    if (msg.role === "assistant" && msg.tool_calls?.length) {
      const calls = msg.tool_calls
        .map(
          (tc) =>
            `<TOOL_CALL>\ntool: ${tc.function.name}\nparams:\n${tc.function.arguments}\n</TOOL_CALL>`,
        )
        .join("\n\n");
      const flat = flattenContent(msg.content).trim();
      content = [flat, calls].filter(Boolean).join("\n");
    } else if (msg.role === "tool") {
      const result = compactSearchToolResult(
        flattenContent(msg.content).trim(),
        msg.name,
      );
      content = `<TOOL_RESULT>\n${result || "(no result returned)"}\n</TOOL_RESULT>\nUse this result to advance the original task. You may call another listed tool for missing context; do not repeat the same search query.`;
    } else {
      const flat = flattenContent(msg.content);
      content =
        Array.isArray(msg.content) &&
        msg.content.some((part) => part.type === "image_url")
          ? msg.content
          : flat.length > 0
            ? flat
            : typeof msg.content === "string"
              ? msg.content
              : "";
    }

    return [{ role, content }];
  });

  // chatplayground has no assistant prefill and no reasoning tags. A trailing
  // assistant draft, or a <think>/<tool_selection> block, is echoed or rejected
  // instead of producing a tool call. Move that text into the last user turn.
  const prepared: UpstreamMessage[] = [];
  for (const message of messages) {
    if (message.role !== "assistant" || typeof message.content !== "string") {
      prepared.push(message);
      continue;
    }
    // Keep tool-call history as-is. Only fold free-form drafts that would
    // otherwise leave the request ending on an assistant turn.
    if (/<TOOL_CALL>/i.test(message.content)) {
      prepared.push(message);
      continue;
    }
    const plain = message.content
      .replace(/<\/?think>/gi, "")
      .replace(/<\/?tool_selection>/gi, "")
      .replace(/\*\*ANSWER\*\*/g, "")
      .trim();
    if (!plain) continue;
    const previous = prepared.at(-1);
    const note = `Draft so far (do not repeat it):\n${plain}`;
    if (
      previous &&
      previous.role === "user" &&
      typeof previous.content === "string"
    ) {
      previous.content = `${previous.content}\n\n${note}`;
    } else {
      prepared.push({ role: "user", content: note });
    }
  }
  messages.length = 0;
  messages.push(...prepared);

  const hasToolResult = req.messages.some((message) => message.role === "tool");
  const hasCallerPromptInTranscript = req.messages.some(
    (message) =>
      (message.role === "system" || message.role === "developer") &&
      !flattenContent(message.content).includes("[relay-tool-prompt-v1]"),
  );
  const callerInstructions = req.messages
    .filter(
      (message) =>
        (message.role === "system" || message.role === "developer") &&
        !flattenContent(message.content).includes("[relay-tool-prompt-v1]"),
    )
    .map((message) => flattenContent(message.content).trim())
    .filter(Boolean)
    .join("\n\n");
  // Tool prompting via injectToolPrompt() (chat.ts) only covers "should I
  // call a tool," not "I already have a tool result, now answer." Models
  // observed mimicking the injected "I called X tool with arguments Y"
  // history line instead of synthesizing an answer — seen on both Claude
  // and DeepSeek. This must fire whenever a tool result exists in history,
  // REGARDLESS of whether `tools` is present on this request — compliant
  // clients resend `tools` on every turn, so gating on its absence almost
  // never actually fires in real traffic.
  const hasToolResultInHistory = hasToolResult && !options.promptOnly;
  if (hasToolResultInHistory) {
    const canCallAnotherTool =
      Boolean(req.tools?.length) && req.tool_choice !== "none";
    const listed = (req.tools ?? []).map((tool) => tool.function.name);
    const hasEditTool = listed.some((name) =>
      /edit|replace|write|create|apply|patch|file/i.test(name),
    );
    messages.push({
      role: "system",
      content: canCallAnotherTool
        ? hasEditTool
          ? "A <TOOL_RESULT> is above. Continue the user's task. If another tool is still needed (edit/write/replace/create/search/fetch/terminal), emit one or more <TOOL_CALL> XML blocks now — never Markdown JSON, never a prose plan. Do not repeat an identical search query. Otherwise answer from the results."
          : "Use every <TOOL_RESULT>. If more evidence or actions are needed, emit <TOOL_CALL> XML for listed tools (including web_search/web_fetch/create tools); never Markdown JSON and don't repeat an identical search query. Otherwise answer."
        : "You stopped calling tools. Answer the user's question from every <TOOL_RESULT> above. Do not announce another search.",
    });
    if (callerInstructions && !hasCallerPromptInTranscript) {
      messages.push({
        role: options.promptOnly || options.contextFilesOnly ? "user" : "system",
        content: `Caller instructions (highest priority; continue to follow them):\n${callerInstructions}`,
      });
    }
  }

  // Hard checks last, after the result note. Upstream rejects an
  // assistant-final turn and an empty message list.
  const last = messages.at(-1);
  if (!options.promptOnly && last?.role === "assistant") {
    const canCallAnotherTool = Boolean(req.tools?.length);
    messages.push({
      role: "user",
      content: canCallAnotherTool
        ? "Continue. If another listed tool is needed to finish the task, emit <TOOL_CALL> XML now (no JSON fence); otherwise answer."
        : "Continue.",
    });
  }
  if (messages.length === 0) {
    messages.push({ role: "user", content: "Hello." });
  }

  // Upstream accepts one message; fold the transcript in caller order.
  const upstreamMessages =
    options.preserveRoles && !options.promptOnly
      ? messages
          .filter(
            (message) =>
              message.role === "system" || message.role === "user",
          )
          .map((message) => ({
            ...message,
            content: boundTranscript(message.content),
          }))
      : collapseHistory(messages);

  // OpenAI `metadata.save` extension → !noSave. Default: don't pollute the
  // caller's chatplayground history with API traffic (noSave=true).
  const save = req.metadata?.save ?? false;

  // Fields shared by all three endpoints.
  const base = {
    messages: upstreamMessages,
    // Resume only from a chat ID cached for this relay conversation.
    chatId: options.chatId ?? "",
    isRegenerate: false,
    promptTemplate: null,
    fileUrl: options.fileUrls?.slice(0, 6).join("|||") ?? null,
    botId: model.upstreamBotId,
    submissionId: crypto.randomUUID(),
    // Keep upstream chats unsaved; each request includes its own transcript.
    noSave: !save,
    ...(req.reasoning_effort ? { reasoning_effort: req.reasoning_effort } : {}),
    ...(req.verbosity ? { verbosity: req.verbosity } : {}),
    ...(req.max_tokens !== undefined ? { maxTokens: req.max_tokens } : {}),
  };

  // The model identifier field differs per endpoint (see types/upstream.ts):
  // azure wants the provider/model slug; perplexity wants a bare modelName;
  // lmsys wants the bare name in `model`. perplexity/lmsys also take an
  // apiKey — null means "use chatplayground's own upstream key" (the relay
  // is BYO-less, so always null).
  const endpoint = model.endpoint;
  switch (endpoint) {
    case "azure":
      return { endpoint, body: { ...base, model: model.upstreamModel } };
    case "perplexity":
      return {
        endpoint,
        body: { ...base, modelName: model.modelName, apiKey: null },
      };
    case "lmsys":
      return {
        endpoint,
        body: { ...base, model: model.modelName, apiKey: null },
      };
    case "image":
      return {
        endpoint,
        body: { ...base, model: model.modelName, apiKey: null },
      };
  }
}

const ROLE_LABEL: Record<UpstreamMessage["role"], string> = {
  system: "System",
  user: "User",
  assistant: "Assistant",
};

/** Collapse the local transcript to the single message accepted upstream. */
function collapseHistory(messages: UpstreamMessage[]): UpstreamMessage[] {
  if (messages.length === 1) {
    const only = messages[0];
    if (!only) return [{ role: "user", content: "" }];
    const content = singleMessageContent(only);
    if (typeof content === "string" && content.startsWith("<TOOL_RESULT>")) {
      return [{ role: "user", content: `Tool: ${content}` }];
    }
    return [{ role: "user", content }];
  }

  const transcriptMessages = messages.filter(
    (message) => !(message.role === "assistant" && message.content === ""),
  );
  if (transcriptMessages.length === 1) {
    const only = transcriptMessages[0];
    return only
      ? [{ role: "user", content: singleMessageContent(only) }]
      : [{ role: "user", content: "" }];
  }

  const parts: UpstreamContentPart[] = [];
  const lastUserIndex = transcriptMessages.findLastIndex(
    (message) =>
      message.role === "user" &&
      !(typeof message.content === "string" &&
        message.content.startsWith("<TOOL_RESULT>")),
  );
  const toolPromptIndex = transcriptMessages.findIndex(
    (message) =>
      message.role === "system" &&
      typeof message.content === "string" &&
      message.content.includes("[relay-tool-prompt-v1]"),
  );
  const orderedMessages =
    toolPromptIndex >= 0 && lastUserIndex >= 0
      ? [
          transcriptMessages[lastUserIndex]!,
          ...transcriptMessages.filter((_, index) => index !== lastUserIndex),
        ]
      : transcriptMessages;
  for (const [index, message] of orderedMessages.entries()) {
    if (index > 0) parts.push({ type: "text", text: "\n\n" });
    const label =
      message.role === "user" &&
      typeof message.content === "string" &&
      message.content.startsWith("<TOOL_RESULT>")
        ? "Tool"
        : ROLE_LABEL[message.role];
    if (typeof message.content === "string") {
      parts.push({ type: "text", text: `${label}: ${message.content}` });
    } else {
      parts.push({ type: "text", text: `${label}:` }, ...message.content);
    }
  }

  const textParts = parts.filter(
    (part): part is UpstreamContentPartText => part.type === "text",
  );
  const transcript: string | UpstreamContentPart[] =
    textParts.length === parts.length
      ? textParts.map((part) => part.text).join("")
      : parts;
  return [
    {
      role: "user",
      content: boundTranscript(transcript),
    },
  ];
}

/** Preserve a lone message's original payload without adding a transcript label. */
function singleMessageContent(
  message: UpstreamMessage,
): UpstreamMessage["content"] {
  if (typeof message.content === "string") {
    return boundTranscript(message.content) as string;
  }
  return boundTranscript(message.content);
}

/** Keep the newest context within the upstream per-message size limit. */
const TOOL_POLICY_MARK = "=== TOOL POLICY ===";
const TOOL_POLICY_END_MARK = "</AVAILABLE_TOOLS>";

/** Keep system/tool definitions intact; trim only older conversation history. */
function preserveToolPolicy(content: string, bounded: string): string {
  const mark = content.indexOf(TOOL_POLICY_MARK);
  if (mark < 0) return bounded;
  const userStart = content.lastIndexOf("\n\nUser: ");
  if (userStart < 0) return bounded;
  const latestUser = content.slice(userStart + 2);
  const end = content.indexOf(TOOL_POLICY_END_MARK, mark);
  const policyEnd =
    end < 0 ? content.length : end + TOOL_POLICY_END_MARK.length;
  const systemPrefix = content.slice(0, mark).trim();
  const policy = content.slice(mark, policyEnd);
  const protectedPrompt = [systemPrefix, policy].filter(Boolean).join("\n\n");
  const preservedUser = latestUser.slice(
    -Math.floor(MAX_UPSTREAM_MESSAGE_CHARS / 2),
  );
  const availablePrompt = Math.max(
    0,
    MAX_UPSTREAM_MESSAGE_CHARS -
      preservedUser.length -
      OMITTED_CONTEXT.length -
      4,
  );
  const prompt = protectedPrompt.slice(0, availablePrompt);
  return `${preservedUser}\n\n${OMITTED_CONTEXT}${prompt}`;
}

function boundTranscript(
  content: string | UpstreamContentPart[],
): string | UpstreamContentPart[] {
  if (typeof content === "string") {
    if (content.length <= MAX_UPSTREAM_MESSAGE_CHARS) return content;
    let start =
      content.length - (MAX_UPSTREAM_MESSAGE_CHARS - OMITTED_CONTEXT.length);
    const firstCode = content.charCodeAt(start);
    if (firstCode >= 0xdc00 && firstCode <= 0xdfff) start++;
    return preserveToolPolicy(content, `${OMITTED_CONTEXT}${content.slice(start)}`);
  }

  let remaining = MAX_UPSTREAM_MESSAGE_CHARS - OMITTED_CONTEXT.length;
  let truncated = false;
  const kept: UpstreamContentPart[] = [];
  let i = content.length - 1;
  for (; i >= 0 && remaining > 0; i--) {
    const part = content[i];
    if (!part) continue;
    if (part.type === "text") {
      let text = part.text;
      if (text.length > remaining) {
        truncated = true;
        text = text.slice(text.length - remaining);
        const firstCode = text.charCodeAt(0);
        if (firstCode >= 0xdc00 && firstCode <= 0xdfff) text = text.slice(1);
      }
      kept.push({ type: "text", text });
      remaining -= text.length;
    } else {
      const size = JSON.stringify(part).length;
      if (size <= remaining) {
        kept.push(part);
        remaining -= size;
      } else {
        truncated = true;
      }
    }
  }
  if (i >= 0) truncated = true;
  const ordered = kept.reverse();
  if (truncated) {
    if (ordered[0]?.type === "text") {
      ordered[0] = {
        type: "text",
        text: `${OMITTED_CONTEXT}${ordered[0].text}`,
      };
    } else {
      ordered.unshift({ type: "text", text: OMITTED_CONTEXT });
    }
  }
  return ordered;
}

export function endpointUrl(
  endpoint: UpstreamEndpoint,
  baseChatUrl: string,
): string {
  // The three chat endpoints are siblings under /api/chat/. Resolving the
  // endpoint name relative to the configured azure URL yields the others, so
  // a single UPSTREAM_CHAT_URL var repoints the whole set at one instance.
  return new URL(endpoint, baseChatUrl).toString();
}

export interface UpstreamHeaderEnv {
  UPSTREAM_ORIGIN: string;
  UPSTREAM_REFERER: string;
}

export function buildUpstreamHeaders(
  sessionToken: string,
  env: UpstreamHeaderEnv,
): HeadersInit {
  // Match the headers observed on the successful chatplayground browser call.
  return {
    accept: "*/*",
    "accept-language": "en-US,en;q=0.9",
    authorization: `Bearer ${sessionToken}`,
    "content-type": "application/json",
    dnt: "1",
    origin: env.UPSTREAM_ORIGIN,
    priority: "u=1, i",
    referer: env.UPSTREAM_REFERER,
    "sec-ch-ua": '"Brave";v="155", "Chromium";v="155", "Not(A:Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1",
    "user-agent":
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36",
  };
}

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

const MAX_UPSTREAM_MESSAGE_CHARS = 15_000;

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

export interface BuiltUpstreamRequest {
  endpoint: UpstreamEndpoint;
  body: UpstreamChatRequest;
}

/** Agent endpoints reject transcripts and accept only the current prompt. */
export function requiresNewPromptOnly(model: ModelEntry): boolean {
  return /\b(?:agent|grok)\b/i.test(
    `${model.id} ${model.modelName} ${model.upstreamModel} ${model.upstreamBotId}`,
  );
}

function latestUserMessage(messages: ChatCompletionRequest["messages"]) {
  return [...messages]
    .toReversed()
    .find((message) => message.role === "user");
}

export function buildUpstreamRequest(
  req: ChatCompletionRequest,
  model: ModelEntry,
  options: { promptOnly?: boolean } = {},
): BuiltUpstreamRequest {
  // chatplayground accepts the same content shape as OpenAI (string or
  // ContentPart[]), so we pass `content` through unchanged. The only
  // role normalization: collapse "tool" → "user" (no endpoint has a tool role).
  //
  // chatplayground also has no concept of `tool_calls`: an assistant
  // message that only carries tool_calls has `content: null`, which the
  // upstream Azure/perplexity/lmsys endpoints reject outright ("400 Invalid
  // value for 'content': expected a string, got null"). Fold tool_calls and
  // tool-role results into plain text instead of dropping/nulling them.
  const promptOnly = options.promptOnly ?? requiresNewPromptOnly(model);
  const sourceMessages = promptOnly || Boolean(req.user)
    ? [latestUserMessage(req.messages) ?? { role: "user" as const, content: "" }]
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
      content = `<TOOL_RESULT>\n${result || "(no result returned)"}\n</TOOL_RESULT>\nTreat this result as evidence for the user's original question. Answer directly with the conclusion first; do not say you cannot verify it or merely list the result. Call another listed tool only when the result is insufficient.`;
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

  // Tool prompting via injectToolPrompt() (chat.ts) only covers "should I
  // call a tool," not "I already have a tool result, now answer." Models
  // observed mimicking the injected "I called X tool with arguments Y"
  // history line instead of synthesizing an answer — seen on both Claude
  // and DeepSeek. This must fire whenever a tool result exists in history,
  // REGARDLESS of whether `tools` is present on this request — compliant
  // clients resend `tools` on every turn, so gating on its absence almost
  // never actually fires in real traffic.
  const hasToolResultInHistory =
    !promptOnly && req.messages.some((m) => m.role === "tool");
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
          ? "A <TOOL_RESULT> is above. Continue the user's task. If an edit/write/replace tool is still needed, emit the next <TOOL_CALL> now. Otherwise answer from the results."
          : "Answer from every <TOOL_RESULT> above. Emit another <TOOL_CALL> only if more information is required."
        : "You stopped calling tools. Answer the user's question from every <TOOL_RESULT> above. Do not announce another search.",
    });
  }

  // Hard checks last, after the result note. Upstream rejects an
  // assistant-final turn and an empty message list.
  const last = messages.at(-1);
  if (last?.role === "assistant") {
    const canCallAnotherTool = Boolean(req.tools?.length);
    messages.push({
      role: "user",
      content: canCallAnotherTool
        ? "Continue. If another listed tool is needed to finish the task, emit a <TOOL_CALL> now; otherwise answer."
        : "Continue.",
    });
  }
  if (messages.length === 0) {
    messages.push({ role: "user", content: "Hello." });
  }

  // chatplayground accepts only one new prompt and relies on chatId for
  // server-side history. This relay stays stateless by folding the OpenAI
  // transcript into one role-labelled user message after local tool and draft
  // handling have preserved their context.
  const upstreamMessages = collapseHistory(messages);

  // OpenAI `metadata.save` extension → !noSave. Default: don't pollute the
  // caller's chatplayground history with API traffic (noSave=true).
  const save = req.metadata?.save ?? false;

  // Fields shared by all three endpoints.
  const base = {
    messages: upstreamMessages,
    // Prompt-only agents resume context by chatId; only their messages array
    // is restricted to the current user prompt.
    chatId: req.user ?? "", // OpenAI `user` → chatplayground `chatId`
    isRegenerate: false,
    promptTemplate: null,
    fileUrl: null,
    botId: model.upstreamBotId,
    submissionId: crypto.randomUUID(),
    // Keep upstream chats unsaved by default. The relay separately caches the
    // returned chatId so follow-up turns can resume without replaying history.
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
    // A lone developer message is still not a valid upstream turn role.
    return [
      {
        role: only.role === "system" ? "system" : "user",
        content: only.content,
      },
    ];
  }

  const parts: UpstreamContentPart[] = [];
  for (const [index, message] of messages.entries()) {
    if (index > 0) parts.push({ type: "text", text: "\n\n" });
    const label =
      message.role === "user" &&
      typeof message.content === "string" &&
      message.content.startsWith("<TOOL_RESULT>")
        ? "Tool"
        : ROLE_LABEL[message.role];
    parts.push({ type: "text", text: `${label}:` });
    if (typeof message.content === "string") {
      parts.push({ type: "text", text: ` ${message.content}` });
    } else {
      parts.push(...message.content);
    }
  }

  const transcript: string | UpstreamContentPart[] = parts.every(
    (part) => part.type === "text",
  )
    ? parts.map((part) => (part as UpstreamContentPartText).text).join("")
    : parts;
  return [{ role: "user", content: boundTranscript(transcript) }];
}

/** Keep the newest context within the upstream per-message size limit. */
function boundTranscript(
  content: string | UpstreamContentPart[],
): string | UpstreamContentPart[] {
  if (typeof content === "string") {
    if (content.length <= MAX_UPSTREAM_MESSAGE_CHARS) return content;
    let start = content.length - MAX_UPSTREAM_MESSAGE_CHARS;
    const firstCode = content.charCodeAt(start);
    if (firstCode >= 0xdc00 && firstCode <= 0xdfff) start++;
    return content.slice(start);
  }

  let remaining = MAX_UPSTREAM_MESSAGE_CHARS;
  const kept: UpstreamContentPart[] = [];
  for (let i = content.length - 1; i >= 0 && remaining > 0; i--) {
    const part = content[i];
    if (!part) continue;
    if (part.type === "text") {
      let text = part.text;
      if (text.length > remaining) {
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
      }
    }
  }
  return kept.reverse();
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

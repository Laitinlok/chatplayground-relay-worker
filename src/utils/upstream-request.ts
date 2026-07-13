import type { UpstreamEndpoint } from "../constants/endpoints";
import type { ModelEntry } from "../constants/models";
import type { ChatCompletionRequest } from "../types/openai";
import type { UpstreamChatRequest, UpstreamMessage } from "../types/upstream";
import { hasTools } from "./tool-shim";


export interface BuiltUpstreamRequest {
  endpoint: UpstreamEndpoint;
  body: UpstreamChatRequest;
}

export function buildUpstreamRequest(
  req: ChatCompletionRequest,
  model: ModelEntry,
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
  const messages: UpstreamMessage[] = req.messages.map((msg) => {
    const role = msg.role === "tool" ? "user" : msg.role;

    function flattenContent(content: unknown): string {
      if (typeof content === "string") return content;
      if (Array.isArray(content)) {
        return content
          .filter((p: any) => p?.type === "text")
          .map((p: any) => p.text)
          .join("\n");
      }
      return "";
    }

    if (msg.role === "assistant" && msg.tool_calls?.length) {
      const calls = msg.tool_calls
        .map((tc) => `I called the "${tc.function.name}" tool with arguments ${tc.function.arguments}.`)
        .join(" ");
      const flat = flattenContent(msg.content).trim();
      return { role, content: flat.length > 0 ? flat : calls };
    }

    if (msg.role === "tool") {
      const result = flattenContent(msg.content).trim();
      return {
        role,
        content: `Here is the result of that tool call:\n${result || "(no result returned)"}\n\nUse this information to answer the user's original question in natural, conversational language. Do not just repeat the tool call or result verbatim.`,
      };
    }

    const flat = flattenContent(msg.content);
    return { role, content: flat.length > 0 ? flat : (typeof msg.content === "string" ? msg.content : "") };
  });

  // Tool prompting via injectToolPrompt() (chat.ts) only covers "should I
  // call a tool," not "I already have a tool result, now answer." Models
  // observed mimicking the injected "I called X tool with arguments Y"
  // history line instead of synthesizing an answer — seen on both Claude
  // and DeepSeek. This must fire whenever a tool result exists in history,
  // REGARDLESS of whether `tools` is present on this request — compliant
  // clients resend `tools` on every turn, so gating on its absence almost
  // never actually fires in real traffic.
  const hasToolResultInHistory = req.messages.some((m) => m.role === "tool");
  if (hasToolResultInHistory) {
    messages.push({
      role: "system",
      content:
        "You already called a tool earlier in this conversation and received its result, shown above. Do not call the tool again and do not restate/summarize that you called it. Answer the user's original question now, directly, using that result.",
    });
  }  

  // OpenAI `metadata.save` extension → !noSave. Default: don't pollute the
  // caller's chatplayground history with API traffic (noSave=true).
  const save = req.metadata?.save ?? false;

  // Fields shared by all three endpoints.
  const base = {
    messages,
    chatId: req.user ?? "", // OpenAI `user` → chatplayground `chatId`
    isRegenerate: false,
    promptTemplate: null,
    fileUrl: null,
    botId: model.upstreamBotId,
    noSave: !save,
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
  }
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
  clerkUserId: string,
  env: UpstreamHeaderEnv,
): HeadersInit {
  // text/plain bypasses CORS preflight — chatplayground's frontend uses this
  // exact content-type and the backend enforces it.
  return {
    "content-type": "text/plain;charset=UTF-8",
    "x-clerk-user-id": clerkUserId,
    origin: env.UPSTREAM_ORIGIN,
    referer: env.UPSTREAM_REFERER,
    accept: "*/*",
  };
}

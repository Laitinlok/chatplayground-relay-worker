import type { OpenAIMessage } from "../types/openai";

export interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: unknown;
  };
}

export type ToolChoice =
  | "none"
  | "auto"
  | { type: "function"; function: { name: string } };

export interface ShimToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

const ENVELOPE_KEY = "relay_tool_call";

export function hasTools(tools: unknown): tools is OpenAITool[] {
  return Array.isArray(tools) && tools.length > 0;
}

function toolChoiceName(toolChoice?: ToolChoice): string | null {
  if (typeof toolChoice === "object" && toolChoice?.type === "function") {
    return toolChoice.function.name;
  }
  return null;
}

export function buildToolSystemPrompt(
  tools: OpenAITool[],
  toolChoice?: ToolChoice,
): string {
  const forcedName = toolChoiceName(toolChoice);

  const policy =
    toolChoice === "none"
      ? "You must not call any tool. Answer normally in plain text."
      : forcedName
        ? `You must call exactly one tool named "${forcedName}".`
        : "If a tool is needed to answer, call exactly one tool. Otherwise answer normally in plain text.";

  const catalog = tools.map((t) => ({
    name: t.function.name,
    description: t.function.description ?? "",
    parameters: t.function.parameters ?? { type: "object", properties: {} },
  }));

  return [
    "You are running behind an OpenAI-compatible relay that has no native tool-calling support.",
    "When you decide to call a tool, respond with ONLY the following minified JSON and nothing else:",
    `{"${ENVELOPE_KEY}":{"name":"<tool_name>","arguments":{...}}}`,
    "Do not wrap the JSON in markdown code fences. Do not add explanation before or after it.",
    "If you are not calling a tool, respond with normal plain-text prose as usual.",
    policy,
    `Available tools: ${JSON.stringify(catalog)}`,
  ].join("\n");
}

export function injectToolPrompt(
  messages: OpenAIMessage[],
  tools: OpenAITool[],
  toolChoice?: ToolChoice,
): OpenAIMessage[] {
  const toolPrompt = buildToolSystemPrompt(tools, toolChoice);
  const [first, ...rest] = messages;

  if (first?.role === "system" && typeof first.content === "string") {
    return [{ ...first, content: `${first.content}\n\n${toolPrompt}` }, ...rest];
  }

  return [{ role: "system", content: toolPrompt }, ...messages];
}

/**
 * Attempts to parse a full assistant reply as a relay tool-call envelope.
 * Returns null if the text isn't a well-formed envelope — callers should
 * then treat the text as ordinary prose.
 */
export function tryParseRelayToolCall(text: string): ShimToolCall | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return null;
  }

  const envelope = (parsed as Record<string, unknown>)?.[ENVELOPE_KEY] as
    | { name?: unknown; arguments?: unknown }
    | undefined;

  if (!envelope || typeof envelope.name !== "string" || typeof envelope.arguments !== "object") {
    return null;
  }

  return {
    id: `call_${crypto.randomUUID().replace(/-/g, "")}`,
    type: "function",
    function: {
      name: envelope.name,
      arguments: JSON.stringify(envelope.arguments ?? {}),
    },
  };
}

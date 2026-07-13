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
const TOOL_CALL_EXAMPLES = [
  `{"${ENVELOPE_KEY}":{"name":"web_search","arguments":{"query":"latest TikTok food hacks","num_results":5}}}`,
];

// Anthropic-style native dialect, observed from Claude regardless of the
// relay envelope instruction. Kept as a second accepted example so the
// model is told this is fine to use, rather than something to avoid.
const NATIVE_DIALECT_EXAMPLES = [
  `<function_calls>\n<invoke name="web_search">\n<parameter name="query">latest TikTok food hacks</parameter>\n</invoke>\n</function_calls>`,
];

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
    "When you decide to call a tool, use whichever tool-calling format you were natively trained on —",
    "your own XML-tag dialect, JSON envelope, or function-call syntax are all fine. The relay will",
    "parse whatever format you naturally produce.",
    "If you have no strong native preference, fall back to this minified JSON with nothing else in the reply:",
    `{"${ENVELOPE_KEY}":{"name":"","arguments":{...}}}`,
    "Do not wrap the tool call in markdown code fences. Do not add explanation before or after it.",
    `Relay-envelope example: ${TOOL_CALL_EXAMPLES[0]}`,
    `Native-dialect example (also acceptable): ${NATIVE_DIALECT_EXAMPLES[0]}`,
    "Call exactly one tool per turn — do not emit a second tool call or any further prose in the same reply",
    "after the first one, even to explain what you're about to do next.",
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

const INVOKE_BLOCK_RE = /<invoke\s+name="([^"]+)"\s*>([\s\S]*?)<\/invoke>/i;
const PARAM_RE = /<parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>([\s\S]*?)<\/parameter>/gi;
const CLAUDE_TOOL_CALL_TAG_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/i;

function coerceParamValue(raw: string, stringFlag: string | undefined): unknown {
  if (stringFlag === "true") return raw;
  const trimmed = raw.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
}

/** Anthropic native `<invoke name="...">…</invoke>` dialect. */
function parseInvokeDialect(text: string): { name: string; arguments: Record<string, unknown> } | null {
  const match = INVOKE_BLOCK_RE.exec(text);
  if (!match) return null;
  const [, name, body] = match;
  const args: Record<string, unknown> = {};
  let paramMatch: RegExpExecArray | null;
  PARAM_RE.lastIndex = 0;
  while ((paramMatch = PARAM_RE.exec(body)) !== null) {
    const [, paramName, stringFlag, paramValue] = paramMatch;
    args[paramName] = coerceParamValue(paramValue, stringFlag);
  }
  return { name, arguments: args };
}

/** Bare `<tool_call>{"name":...,"arguments":{...}}</tool_call>` dialect. */
function parseToolCallTagDialect(text: string): { name: string; arguments: Record<string, unknown> } | null {
  const match = CLAUDE_TOOL_CALL_TAG_RE.exec(text);
  if (!match?.[1]) return null;
  try {
    const obj = JSON.parse(match[1]);
    if (typeof obj?.name !== "string") return null;
    return { name: obj.name, arguments: extractArguments(obj) };
  } catch {
    return null;
  }
}


function extractArguments(obj: Record<string, unknown>): Record<string, unknown> {
  const nested = obj.arguments;
  if (nested && typeof nested === "object" && !Array.isArray(nested)) {
    return nested as Record<string, unknown>;
  }
  const { name: _name, arguments: _arguments, ...rest } = obj;
  return rest;
}

function extractBalancedJson(buf: string, startIdx: number): string | null {
  if (startIdx < 0 || buf[startIdx] !== "{") return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = startIdx; i < buf.length; i++) {
    const ch = buf[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === "\"") {
        inString = false;
      }
      continue;
    }
    if (ch === "\"") {
      inString = true;
      continue;
    }
    if (ch === "{") depth++;
    if (ch === "}") {
      depth--;
      if (depth === 0) return buf.slice(startIdx, i + 1);
    }
  }
  return null;
}

function findEnvelopeJson(text: string): string | null {
  const keyIdx = text.indexOf(`"${ENVELOPE_KEY}"`);
  if (keyIdx === -1) return null;
  const openIdx = text.lastIndexOf("{", keyIdx);
  if (openIdx === -1) return null;
  return extractBalancedJson(text, openIdx);
}

/**
 * Attempts to parse a full assistant reply as a relay tool-call envelope.
 * Returns null if the text isn't a well-formed envelope — callers should
 * then treat the text as ordinary prose.
 */
export function tryParseRelayToolCall(text: string): ShimToolCall | null {
  const trimmed = text.trim();

  // Native dialect fallbacks — checked first since a model committing to
  // its own trained syntax is the expected, encouraged path now.
  const invoke = parseInvokeDialect(trimmed);
  if (invoke) {
    return {
      id: `call_${crypto.randomUUID().replace(/-/g, "")}`,
      type: "function",
      function: { name: invoke.name, arguments: JSON.stringify(invoke.arguments) },
    };
  }
  const tagged = parseToolCallTagDialect(trimmed);
  if (tagged) {
    return {
      id: `call_${crypto.randomUUID().replace(/-/g, "")}`,
      type: "function",
      function: { name: tagged.name, arguments: JSON.stringify(tagged.arguments) },
    };
  }

  const envelopeJson = findEnvelopeJson(trimmed) ?? (trimmed.startsWith("{") ? trimmed : null);
  if (!envelopeJson) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(envelopeJson);
  } catch {
    return null;
  }

  const wrapper = parsed as Record<string, unknown>;
  const rawEnvelope = (wrapper?.[ENVELOPE_KEY] ?? wrapper) as Record<string, unknown> | undefined;
  if (!rawEnvelope || typeof rawEnvelope.name !== "string") return null;
  const args = extractArguments(rawEnvelope);

  return {
    id: `call_${crypto.randomUUID().replace(/-/g, "")}`,
    type: "function",
    function: {
      name: rawEnvelope.name,
      arguments: JSON.stringify(args),
    },
  };
}

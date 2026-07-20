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

export function hasTools(tools: unknown): tools is OpenAITool[] {
  return Array.isArray(tools) && tools.length > 0 && tools.every((tool) => {
    if (!tool || typeof tool !== "object") return false;
    const candidate = tool as Record<string, unknown>;
    const fn = candidate.function;
    return candidate.type === "function"
      && !!fn
      && typeof fn === "object"
      && typeof (fn as Record<string, unknown>).name === "string"
      && ((fn as Record<string, unknown>).name as string).trim().length > 0;
  });
}

interface ParsedToolIntent {
  name: string;
  arguments: Record<string, unknown>;
}

const ENVELOPE_KEY = "relay_tool_call";
const ENVELOPE_KEY_ALIASES = [
  "relay_tool_call",
  "tool_call",
  "function_call",
  "call",
  "invoke",
];
const NAME_KEY_ALIASES = ["name", "tool", "tool_name", "function", "function_name"];
const ARGUMENT_KEY_ALIASES = ["arguments", "args", "parameters", "params", "input"];
const TOOL_NAME_ALIASES: Record<string, string[]> = {
  cron: ["schedule", "scheduler", "reminder", "create_reminder", "cron_add"],
};
const TOOL_CALL_EXAMPLES = [
  `TOOL_CALL: web_search\nARGUMENTS: {"query":"latest TikTok food hacks","num_results":5}`,
  `{"${ENVELOPE_KEY}":{"name":"web_search","arguments":{"query":"latest TikTok food hacks","num_results":5}}}`,
];

function formatToolCatalog(tools: OpenAITool[]): string {
  return tools.map((tool) => {
    const schema = tool.function.parameters;
    if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
      return `- ${tool.function.name}: ${tool.function.description ?? ""}`;
    }
    const record = schema as Record<string, unknown>;
    const required = Array.isArray(record.required) ? record.required : [];
    const properties = record.properties && typeof record.properties === "object"
      ? Object.entries(record.properties as Record<string, unknown>)
      : [];
    const params = properties.map(([name, value]) => {
      const property = value && typeof value === "object" ? value as Record<string, unknown> : {};
      const marker = required.includes(name) ? " (required)" : " (optional)";
      return `  - ${name}${marker}: ${String(property.description ?? property.type ?? "value")}`;
    });
    return [`- ${tool.function.name}: ${tool.function.description ?? ""}`, ...params].join("\n");
  }).join("\n");
}

function normalizeName(value: string): string {
  return value
    .trim()
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_+/g, "_");
}

function editDistance(a: string, b: string): number {
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let diagonal = row[0]!;
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const previous = row[j]!;
      row[j] = a[i - 1] === b[j - 1]
        ? diagonal
        : 1 + Math.min(diagonal, row[j]!, row[j - 1]!);
      diagonal = previous;
    }
  }
  return row[b.length]!;
}

function toolNameDistance(a: string, b: string): number {
  let distance = editDistance(a, b);
  for (let i = 0; i < a.length - 1; i++) {
    const chars = a.split("");
    [chars[i], chars[i + 1]] = [chars[i + 1]!, chars[i]!];
    distance = Math.min(distance, editDistance(chars.join(""), b));
  }
  return distance;
}

function resolveToolName(requested: string, tools?: OpenAITool[]): string | null {
  if (!tools?.length) return requested.trim();
  const normalized = normalizeName(requested);
  const candidates = tools.map((tool) => ({
    name: tool.function.name,
    normalized: normalizeName(tool.function.name),
  }));
  const exact = candidates.find((candidate) => candidate.normalized === normalized);
  if (exact) return exact.name;

  const alias = candidates.find((candidate) =>
    (TOOL_NAME_ALIASES[candidate.normalized] ?? []).some((value) => normalizeName(value) === normalized),
  );
  if (alias) return alias.name;

  const scored = candidates
    .map((candidate) => {
      const distance = toolNameDistance(normalized, candidate.normalized);
      const scale = Math.max(normalized.length, candidate.normalized.length, 1);
      return { ...candidate, score: 1 - distance / scale };
    })
    .sort((a, b) => b.score - a.score);
  const best = scored[0];
  const second = scored[1];
  // Require both a strong match and separation from the runner-up.
  if (best && best.score >= 0.70 && (!second || best.score - second.score >= 0.08)) {
    return best.name;
  }
  return null;
}

function parseArgumentValue(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value === "string") return parseJsonObject(value);
  return null;
}

function findValue(obj: Record<string, unknown>, aliases: string[]): unknown {
  const entry = Object.entries(obj).find(([key]) => aliases.includes(normalizeName(key)));
  return entry?.[1];
}

function buildIntent(obj: Record<string, unknown>): ParsedToolIntent | null {
  const name = findValue(obj, NAME_KEY_ALIASES);
  if (typeof name !== "string" || !name.trim()) return null;
  const rawArgs = findValue(obj, ARGUMENT_KEY_ALIASES);
  const args = rawArgs === undefined ? extractArguments(obj) : parseArgumentValue(rawArgs);
  return { name, arguments: args ?? {} };
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
        : "If a tool is needed to answer accurately or to complete a multi-step task, call it — do not answer from memory when a tool exists that would give a more current or verified result (e.g. translation, unit conversion via calculator, or live data). Multi-step tasks may require several tool calls across turns, one call per turn, in sequence.";

  const catalog = tools.map((t) => ({
    name: t.function.name,
    description: t.function.description ?? "",
    parameters: t.function.parameters ?? { type: "object", properties: {} },
  }));

  return [
    "You are running behind an OpenAI-compatible relay that has no native tool-calling support.",
    "Available tools:",
    formatToolCatalog(tools),
    "Only use tools from this list; never invent a tool name.",
    "When calling a tool, emit exactly one call using this format:",
    `TOOL_CALL: <tool name>\nARGUMENTS: <valid JSON object>`,
    `Example:\n${TOOL_CALL_EXAMPLES[0]}`,
    "Include every required parameter. Make one call at a time and wait for the tool result.",
    "After a tool result is provided, either answer the user directly or make the next necessary call.",
    "Do not wrap a tool call in markdown fences and do not add prose after its arguments.",
    `JSON envelope fallback (also accepted): ${TOOL_CALL_EXAMPLES[1]}`,
    "Call exactly one tool per turn — do not emit a second tool call or any further prose in the same reply",
    "after the first one, even to explain what you're about to do next. It is expected and correct to make",
    "additional tool calls on later turns if the task requires more than one step (e.g. look up a contact,",
    "then create a calendar event; search for a file, then read it, then send its contents).",
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
const TEXT_TOOL_CALL_RE = /(?:^|\n)\s*TOOL_CALL\s*:\s*([^\n]+)\s*\n\s*ARGUMENTS\s*:\s*/i;
const NATIVE_JSON_TOOL_CALL_RE = /"tool_calls"\s*:\s*\[/i;
const PROSE_TOOL_CALL_RE = /\bI\s+(?:called|call|am calling|will call)\s+the\s+"([^"]+)"\s+tool\s+with\s+arguments\s*/i;

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
  const name = match[1] ?? "";
  const body = match[2] ?? "";
  if (!name || name.trim() === "") return null;
  const args: Record<string, unknown> = {};
  let paramMatch: RegExpExecArray | null;
  PARAM_RE.lastIndex = 0;
  while ((paramMatch = PARAM_RE.exec(body)) !== null) {
    const [, paramName = "", stringFlag, paramValue = ""] = paramMatch;
    if (paramName) args[paramName] = coerceParamValue(paramValue, stringFlag);
  }
  return { name, arguments: args };
}

/** Bare `<tool_call>{"name":...,"arguments":{...}}</tool_call>` dialect. */
function parseToolCallTagDialect(text: string): ParsedToolIntent | null {
  const match = CLAUDE_TOOL_CALL_TAG_RE.exec(text);
  if (!match?.[1]) return null;
  try {
    const obj = JSON.parse(match[1]);
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) return null;
    return buildIntent(obj as Record<string, unknown>);
  } catch {
    return null;
  }
}

/**
 * OpenAI-compatible relays sometimes receive a model's natural-language
 * narration of a tool call instead of the machine-readable call itself, e.g.
 * `I called the "web_search" tool with arguments {...}.`. Treat that as a
 * recoverable tool-call dialect so clients such as OpenClaw never see the
 * malformed prose as assistant content.
 */
function parseTextToolCallDialect(text: string): ParsedToolIntent | null {
  const match = TEXT_TOOL_CALL_RE.exec(text);
  if (!match?.[1]) return null;
  const jsonStart = match.index + match[0].length;
  const json = extractBalancedJson(text, text.indexOf("{", jsonStart));
  if (!json) return null;
  const args = parseJsonObject(json) ?? parseLooseJsonObject(json);
  if (!args) return null;
  return { name: match[1].trim(), arguments: args };
}

function parseNativeJsonToolCallDialect(text: string): ParsedToolIntent | null {
  if (!NATIVE_JSON_TOOL_CALL_RE.test(text)) return null;
  const nameMatch = /"name"\s*:\s*"([^"]+)"/.exec(text);
  if (!nameMatch) return null;
  const argumentsMatch = /"arguments"\s*:\s*("(?:\\.|[^"\\])*"|\{)/.exec(text);
  if (!argumentsMatch) return null;
  if (argumentsMatch[1] === "{") {
    const args = extractBalancedJson(text, argumentsMatch.index + argumentsMatch[0].length - 1);
    return args ? { name: nameMatch[1]!, arguments: parseJsonObject(args) ?? {} } : null;
  }
  try {
    const decoded = JSON.parse(argumentsMatch[1]!);
    const args = parseArgumentValue(decoded);
    return args ? { name: nameMatch[1]!, arguments: args } : null;
  } catch {
    return null;
  }
}

function parseProseToolCallDialect(text: string): ParsedToolIntent | null {
  const match = PROSE_TOOL_CALL_RE.exec(text);
  if (!match?.[1]) return null;

  const jsonStart = text.indexOf("{", match.index + match[0].length);
  const json = extractBalancedJson(text, jsonStart);
  if (!json) return null;

  const parsed = parseJsonObject(json) ?? parseLooseJsonObject(json);
  if (!parsed) return null;

    return buildIntent(parsed);
}

function parseJsonObject(json: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(json);
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
  } catch {
    // fall through to caller fallback
  }
  return null;
}

/**
 * Best-effort recovery for the common malformed search-query case where a
 * quoted query is embedded without escaping its internal quotes:
 * `{ "query": ""foo" "bar"", "count": 10 }`.
 * This is intentionally narrow: it only repairs string values by escaping
 * interior quotes between a property colon and the next comma/end brace.
 */
function parseLooseJsonObject(json: string): Record<string, unknown> | null {
  const repaired = json.replace(
    /(:\s*")([\s\S]*?)("\s*(?=,\s*"[A-Za-z0-9_$-]+"\s*:|\s*}))/g,
    (_full: string, prefix: string, value: string, suffix: string) =>
      prefix + value.replace(/(?<!\\)"/g, '\\"') + suffix,
  );
  return repaired === json ? null : parseJsonObject(repaired);
}

function extractArguments(obj: Record<string, unknown>): Record<string, unknown> {
  const nested = obj.arguments;
  if (nested && typeof nested === "object" && !Array.isArray(nested)) {
    return nested as Record<string, unknown>;
  }

  // Some models (observed inconsistently across dialects) emit `arguments`
  // as a JSON-encoded string rather than a nested object — e.g.
  // "arguments":"{\"filepath\":\"foo.py\"}" instead of "arguments":{"filepath":"foo.py"}.
  // Without this branch, the check above fails silently and the rest-spread
  // below returns {} since name/arguments are typically the only two keys
  // present — producing a "successful" tool call with no arguments at all.
  if (typeof nested === "string") {
    try {
      const parsed: unknown = JSON.parse(nested);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // fall through — not valid JSON, treat as no structured args
    }
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
  const keyPattern = /"([^"\\]+)"\s*:/g;
  let match: RegExpExecArray | null;
  while ((match = keyPattern.exec(text)) !== null) {
    if (!ENVELOPE_KEY_ALIASES.some((alias) => normalizeName(match![1]!) === normalizeName(alias))) {
      continue;
    }
    const openIdx = text.lastIndexOf("{", match.index);
    if (openIdx === -1) continue;
    const json = extractBalancedJson(text, openIdx);
    if (json) return json;
  }
  return null;
}

/**
 * Attempts to parse a full assistant reply as a relay tool-call envelope.
 * Returns null if the text isn't a well-formed envelope — callers should
 * then treat the text as ordinary prose.
 */
export function tryParseRelayToolCall(text: string, tools?: OpenAITool[]): ShimToolCall | null {
  const trimmed = text.trim();
  const toToolCall = (intent: ParsedToolIntent | null): ShimToolCall | null => {
    if (!intent) return null;
    const name = resolveToolName(intent.name, tools);
    if (!name) return null;
    return {
      id: `call_${crypto.randomUUID().replace(/-/g, "")}`,
      type: "function",
      function: { name, arguments: JSON.stringify(intent.arguments) },
    };
  };

  // Native dialect fallbacks — checked first since a model committing to
  // its own trained syntax is the expected, encouraged path now.
  const invoke = parseInvokeDialect(trimmed);
  if (invoke) return toToolCall(invoke);
  const tagged = parseToolCallTagDialect(trimmed);
  if (tagged) return toToolCall(tagged);
  const textFormat = parseTextToolCallDialect(trimmed);
  if (textFormat) return toToolCall(textFormat);
  const nativeJson = parseNativeJsonToolCallDialect(trimmed);
  if (nativeJson) return toToolCall(nativeJson);
  const prose = parseProseToolCallDialect(trimmed);
  if (prose) return toToolCall(prose);

  const envelopeJson = findEnvelopeJson(trimmed) ?? (trimmed.startsWith("{") ? trimmed : null);
  if (!envelopeJson) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(envelopeJson);
  } catch {
    parsed = parseLooseJsonObject(envelopeJson);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;

  const wrapper = parsed as Record<string, unknown>;
  const envelope = findValue(wrapper, ENVELOPE_KEY_ALIASES);
  const intent = parseArgumentValue(envelope)
    ? buildIntent(parseArgumentValue(envelope)!)
    : buildIntent(wrapper);
  return toToolCall(intent);
}

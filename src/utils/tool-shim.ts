import type { OpenAIMessage } from "../types/openai";

export interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: unknown;
  };
}

interface FlatFunctionTool {
  type: "function";
  name: string;
  description?: string;
  parameters?: unknown;
}

/** Normalize Chat Completions and Responses-style function tool definitions. */
export function normalizeOpenAITools(tools: unknown): OpenAITool[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((tool): OpenAITool[] => {
    if (!tool || typeof tool !== "object") return [];
    const candidate = tool as Record<string, unknown>;
    if (candidate.type === "web_search_preview" || candidate.type === "web_search") {
      return [
        {
          type: "function",
          function: {
            name: "web_search",
            description:
              "Search the web for a specific query. Choose max_results based on the evidence needed, from 1 to 50.",
            parameters: {
              type: "object",
              properties: {
                query: { type: "string", description: "Focused search query" },
                max_results: {
                  type: "integer",
                  minimum: 1,
                  maximum: 50,
                  description: "Number of relevant results needed",
                },
              },
              required: ["query"],
            },
          },
        },
        {
          type: "function",
          function: {
            name: "web_fetch",
            description:
              "Retrieve the full text for a URL returned by a previous web_search call. Use this to inspect a promising result before answering.",
            parameters: {
              type: "object",
              properties: {
                url: { type: "string", description: "URL from a search result" },
              },
              required: ["url"],
            },
          },
        },
      ];
    }
    if (candidate.type !== "function") return [];

    const nested = candidate.function;
    if (nested && typeof nested === "object") {
      const fn = nested as Record<string, unknown>;
      if (typeof fn.name !== "string" || !fn.name.trim()) return [];
      return [
        {
          type: "function",
          function: {
            name: fn.name,
            ...(typeof fn.description === "string"
              ? { description: fn.description }
              : {}),
            ...(fn.parameters !== undefined
              ? { parameters: fn.parameters }
              : {}),
          },
        },
      ];
    }

    const flat = candidate as Partial<FlatFunctionTool>;
    if (typeof flat.name !== "string" || !flat.name.trim()) return [];
    return [
      {
        type: "function",
        function: {
          name: flat.name,
          ...(typeof flat.description === "string"
            ? { description: flat.description }
            : {}),
          ...(flat.parameters !== undefined
            ? { parameters: flat.parameters }
            : {}),
        },
      },
    ];
  });
}

export function hasTools(tools: unknown): tools is OpenAITool[] {
  return normalizeOpenAITools(tools).length > 0;
}

export type ToolChoice =
  | "none"
  | "auto"
  | "required"
  | { type: "function"; function: { name: string } };

export interface ShimToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

interface ParsedToolIntent {
  name: string;
  arguments: Record<string, unknown>;
}

const ENVELOPE_KEY = "relay_tool_call";
const TOOL_PROMPT_SENTINEL = "[relay-tool-prompt-v1]";
const REASONING_PROMPT_SENTINEL = "[relay-reasoning-prompt-v1]";
const ENVELOPE_KEY_ALIASES = [
  "relay_tool_call",
  "tool_call",
  "function_call",
  "call",
  "invoke",
];
const NAME_KEY_ALIASES = [
  "name",
  "tool",
  "tool_name",
  "function",
  "function_name",
];
const ARGUMENT_KEY_ALIASES = [
  "arguments",
  "args",
  "parameters",
  "params",
  "input",
];
const TOOL_NAME_ALIASES: Record<string, string[]> = {
  cron: ["schedule", "scheduler", "reminder", "create_reminder", "cron_add"],
  web_search: ["web.run", "web_search_preview", "browser.search", "search"],
};
const TOOL_CALL_EXAMPLES = [
  `TOOL_CALL: web_search\nARGUMENTS: {"query":"latest TikTok food hacks","num_results":5}`,
  `{"${ENVELOPE_KEY}":{"name":"web_search","arguments":{"query":"latest TikTok food hacks","num_results":5}}}`,
];

function formatToolCatalog(tools: OpenAITool[]): string {
  return tools
    .map((tool) => {
      const schema = tool.function.parameters;
      if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
        return `- ${tool.function.name}: ${tool.function.description ?? ""}`;
      }
      const record = schema as Record<string, unknown>;
      const required = Array.isArray(record.required) ? record.required : [];
      const properties =
        record.properties && typeof record.properties === "object"
          ? Object.entries(record.properties as Record<string, unknown>)
          : [];
      const params = properties.map(([name, value]) => {
        const property =
          value && typeof value === "object"
            ? (value as Record<string, unknown>)
            : {};
        const marker = required.includes(name) ? " (required)" : " (optional)";
        return `  - ${name}${marker}: ${String(property.description ?? property.type ?? "value")}`;
      });
      return [
        `- ${tool.function.name}: ${tool.function.description ?? ""}`,
        ...params,
      ].join("\n");
    })
    .join("\n");
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
      row[j] =
        a[i - 1] === b[j - 1]
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

function resolveToolName(
  requested: string,
  tools?: OpenAITool[],
): string | null {
  if (!tools?.length) return requested.trim();
  const normalized = normalizeName(requested);
  const candidates = tools.map((tool) => ({
    name: tool.function.name,
    normalized: normalizeName(tool.function.name),
  }));
  const exact = candidates.find(
    (candidate) => candidate.normalized === normalized,
  );
  if (exact) return exact.name;

  const alias = candidates.find((candidate) =>
    (TOOL_NAME_ALIASES[candidate.normalized] ?? []).some(
      (value) => normalizeName(value) === normalized,
    ),
  );
  if (alias) return alias.name;

  const scored = candidates
    .map((candidate) => {
      const distance = toolNameDistance(normalized, candidate.normalized);
      const scale = Math.max(normalized.length, candidate.normalized.length, 1);
      return { ...candidate, score: 1 - distance / scale };
    })
    .toSorted((a, b) => b.score - a.score);
  const best = scored[0];
  const second = scored[1];
  // Require both a strong match and separation from the runner-up.
  if (
    best &&
    best.score >= 0.7 &&
    (!second || best.score - second.score >= 0.08)
  ) {
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
  const entry = Object.entries(obj).find(([key]) =>
    aliases.includes(normalizeName(key)),
  );
  return entry?.[1];
}

function buildIntent(obj: Record<string, unknown>): ParsedToolIntent | null {
  const name = findValue(obj, NAME_KEY_ALIASES);
  if (typeof name !== "string" || !name.trim()) return null;
  const rawArgs = findValue(obj, ARGUMENT_KEY_ALIASES);
  const args =
    rawArgs === undefined ? extractArguments(obj) : parseArgumentValue(rawArgs);
  return { name, arguments: args ?? {} };
}

function toolChoiceName(toolChoice?: ToolChoice): string | null {
  if (typeof toolChoice === "object" && toolChoice?.type === "function") {
    return toolChoice.function.name;
  }
  return null;
}

export type ToolShimProvider = "perplexity";

export function buildToolSystemPrompt(
  tools: OpenAITool[],
  toolChoice?: ToolChoice,
  provider?: ToolShimProvider,
): string {
  const forcedName = toolChoiceName(toolChoice);

  const policy =
    toolChoice === "none"
      ? "You must not call any tool. Answer normally in plain text."
      : toolChoice === "required"
        ? "You must call exactly one tool from the available list."
        : forcedName
          ? `You must call exactly one tool named "${forcedName}".`
          : "If a tool is needed to answer accurately or to complete a multi-step task, call it — do not answer from memory when a tool exists that would give a more current or verified result (e.g. translation, unit conversion via calculator, or live data). Multi-step tasks may require several tool calls across turns, one call per turn, in sequence.";

  const catalog = tools.map((t) => ({
    name: t.function.name,
    description: t.function.description ?? "",
    parameters: t.function.parameters ?? { type: "object", properties: {} },
  }));

  const fileEditTools = tools.filter((tool) =>
    /(?:edit|write|patch|apply|create|delete|read|replace|multi_edit|str_replace|apply_patch).*(?:file|code|project)|(?:file|code|project).*(?:edit|write|patch|apply|create|delete|read|replace)|(?:edit_file|write_file|patch_file|apply_patch|multi_edit|str_replace_editor)/i.test(
      `${tool.function.name} ${tool.function.description ?? ""}`,
    ),
  );
  const decisionRules = [
    ...(fileEditTools.length
      ? [
          "For requests to modify, create, or fix project files, you MUST use an available file/code tool to perform the change directly. Do not answer with a code block, diff, patch text, or instructions instead of invoking the edit tool.",
          "Choose the file tool matching the requested operation, supply its required arguments in the exact schema shown in the tool catalog, and call it immediately. For multi-step edits, read the current file first, then edit, then inspect or test the result. Never claim a change succeeded unless the tool returns success.",
          "A request such as 'fix this', 'change the code', or 'apply this fix' means edit the workspace files when a suitable file-editing tool is available; do not interpret it as a request to merely demonstrate code.",
        ]
      : []),
    ...(tools.some((tool) => tool.function.name === "web_search")
      ? [
          "For web research, choose the search query and max_results needed for the question. Use multiple focused searches when they cover distinct subquestions or you need to verify key claims.",
          "After each web_search result, call web_fetch on at least one relevant returned URL before answering; use more fetches for other promising sources. For multi-part questions or when the first results leave gaps, issue another focused web_search with a meaningfully different query, then fetch relevant links from those results. Continue until the evidence is sufficient, and only then answer. Do not stop after a single search when relevant links are available."
        ]
      : []),
    "If a tool is needed, emit the call immediately; do not preface it with statements such as 'I will search'.",
    "If you are not calling a tool, respond with normal plain-text prose as usual.",
  ];

  const providerRules =
    provider === "perplexity"
      ? [
          "Perplexity compatibility mode is active: the upstream does not receive native tools, so you must act as the tool dispatcher for the host.",
          "For any request that requires a listed tool, do not answer, browse, cite sources, explain your plan, or emit ordinary prose. Emit the tool call immediately.",
          "A tool turn must contain exactly one call in the required TOOL_CALL format. The tool name must match the catalog exactly and ARGUMENTS must be one valid JSON object with every required field.",
          "Never replace a listed host tool with Perplexity's own web search or citations. Never put JSON in a markdown fence.",
          "After the host returns a tool result, use it to continue the task. If another tool is needed, emit another single TOOL_CALL; otherwise answer the user normally.",
        ]
      : [];

  return [
    "You have access to the tools listed below through the relay. The host will execute a tool call and return the result to you.",
    "Never claim that tools are unavailable and never describe this protocol to the user.",
    ...providerRules,
    TOOL_PROMPT_SENTINEL,
    "Available tools:",
    formatToolCatalog(tools),
    "Only use tools from this list; never invent a tool name.",
    "When calling a tool, your entire response must be exactly:",
    `TOOL_CALL: <tool name>\nARGUMENTS: <valid JSON object>`,
    `Example: ${TOOL_CALL_EXAMPLES[0]}`,
    "Do not wrap the call in markdown fences and do not add prose after its arguments.",
    "Include every required parameter. Make one call at a time and wait for the tool result.",
    "After a tool result is provided, either answer the user directly or make the next necessary call.",
    "If no further tool is needed after a tool result, give a complete, self-contained final answer to the user's original request in this turn. Do not stop after a plan, acknowledgement, preamble, or a single incomplete sentence.",
    ...decisionRules,
    "Do not wrap a tool call in markdown fences and do not add prose after its arguments.",
    `Legacy JSON envelope fallback (also accepted): ${TOOL_CALL_EXAMPLES[1]}`,
    "Call exactly one tool per turn — do not emit a second tool call or any further prose in the same reply",
    "after the first one, even to explain what you're about to do next. It is expected and correct to make",
    "additional tool calls on later turns if the task requires more than one step (e.g. look up a contact,",
    "then create a calendar event; search for a file, then read it, then send its contents).",
    policy,
    `Available tools: ${JSON.stringify(catalog)}`,
  ].join("\n");
}

export function injectReasoningPrompt(
  messages: OpenAIMessage[],
  effort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh",
): OpenAIMessage[] {
  if (!effort || effort === "none") return messages;

  const prompt = [
    REASONING_PROMPT_SENTINEL,
    `Reasoning mode is enabled with strength "${effort}".`,
    "Reason carefully before giving the final answer.",
    "Put the reasoning portion inside exactly one <think>...</think> block, followed by the final answer outside that block.",
    "Do not print an unmatched closing </think> tag. Keep the reasoning proportional to the requested strength.",
  ].join("\n");
  const [first, ...rest] = messages;

  if (first?.role === "system" && typeof first.content === "string") {
    if (first.content.includes(REASONING_PROMPT_SENTINEL)) return messages;
    return [{ ...first, content: `${first.content}\n\n${prompt}` }, ...rest];
  }

  return [{ role: "system", content: prompt }, ...messages];
}

export function injectToolPrompt(
  messages: OpenAIMessage[],
  tools: OpenAITool[],
  toolChoice?: ToolChoice,
  provider?: ToolShimProvider,
): OpenAIMessage[] {
  const toolPrompt = buildToolSystemPrompt(tools, toolChoice, provider);
  const [first, ...rest] = messages;

  if (first?.role === "system" && typeof first.content === "string") {
    if (first.content.includes(TOOL_PROMPT_SENTINEL)) return messages;
    return [
      { ...first, content: `${toolPrompt}\n\nSystem Instructions:\n${first.content}` },
      ...rest,
    ];
  }

  return [{ role: "system", content: toolPrompt }, ...messages];
}

const INVOKE_BLOCK_RE = /<invoke\s+name="([^"]+)"\s*>([\s\S]*?)<\/invoke>/i;
const PARAM_RE =
  /<parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>([\s\S]*?)<\/parameter>/gi;
const CLAUDE_TOOL_CALL_TAG_RE = /(?:<tool_call>|<\|tool_call\|>)\s*([\s\S]*?)\s*(?:<\/tool_call>|<\|\/tool_call\|>)/i;
const TEXT_TOOL_CALL_RE =
  /\bTOOL_CALL\s*:\s*([^\r\n]+?)\s+(?:\r?\n\s*)?ARGUMENTS\s*:\s*/i;
const NATIVE_JSON_TOOL_CALL_RE = /"tool_calls"\s*:\s*\[/i;
const PROSE_TOOL_CALL_RE =
  /\bI\s+(?:called|call|am calling|will call)\s+the\s+"([^"]+)"\s+tool\s+with\s+arguments\s*/i;
const RECIPIENT_TOOL_CALL_RE =
  /(?:<\|(?:recipient|channel)\|>|\bto\s*=\s*)(?:(?:functions?|tools?)\.|(?=(?:web\.run|browser\.search)\b))([A-Za-z0-9_.:-]+)[^\n]*?(?:<\|message\|>|<\|constrain\|>json\s*)?/i;

function coerceParamValue(
  raw: string,
  stringFlag: string | undefined,
): unknown {
  if (stringFlag === "true") return raw;
  const trimmed = raw.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
}

/** Anthropic native `<invoke name="...">…</invoke>` dialect. */
function parseInvokeDialect(
  text: string,
): { name: string; arguments: Record<string, unknown> } | null {
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
    const args = extractBalancedJson(
      text,
      argumentsMatch.index + argumentsMatch[0].length - 1,
    );
    return args
      ? { name: nameMatch[1]!, arguments: parseJsonObject(args) ?? {} }
      : null;
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

  return buildIntent({ name: match[1], arguments: parsed });
}

/**
 * GPT-5.x-compatible Harmony/recipient dialect, for example:
 * `<|channel|>commentary to=functions.web_search <|constrain|>json<|message|>{...}`.
 * The recipient is resolved against requested tools before it is exposed.
 */
function parseRecipientToolCallDialect(text: string): ParsedToolIntent | null {
  const match = RECIPIENT_TOOL_CALL_RE.exec(text);
  if (!match?.[1]) return null;
  const json = extractBalancedJson(text, text.indexOf("{", match.index));
  if (!json) return null;
  const args = parseJsonObject(json) ?? parseLooseJsonObject(json);
  return args ? buildIntent({ name: match[1], arguments: args }) : null;
}

function normalizeToolPayload(text: string): string {
  return text
    .trim()
    .replace(/^```(?:json|javascript|js)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
}

function parseSearchArgumentAliases(
  args: Record<string, unknown>,
): Record<string, unknown> {
  if (args.query === undefined && typeof args.search_query === "string") {
    return { ...args, query: args.search_query };
  }
  return args;
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

function extractArguments(
  obj: Record<string, unknown>,
): Record<string, unknown> {
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
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
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
    if (
      !ENVELOPE_KEY_ALIASES.some(
        (alias) => normalizeName(match![1]!) === normalizeName(alias),
      )
    ) {
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
export function tryParseRelayToolCall(
  text: string,
  tools?: OpenAITool[],
): ShimToolCall | null {
  const trimmed = normalizeToolPayload(text);
  const toToolCall = (intent: ParsedToolIntent | null): ShimToolCall | null => {
    if (!intent) return null;
    const name = resolveToolName(intent.name, tools);
    if (!name) return null;
    const argumentsValue =
      normalizeName(name) === "web_search"
        ? parseSearchArgumentAliases(intent.arguments)
        : intent.arguments;
    return {
      id: `call_${crypto.randomUUID().replace(/-/g, "")}`,
      type: "function",
      function: { name, arguments: JSON.stringify(argumentsValue) },
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
  const recipient = parseRecipientToolCallDialect(trimmed);
  if (recipient) return toToolCall(recipient);

  const envelopeJson =
    findEnvelopeJson(trimmed) ?? (trimmed.startsWith("{") ? trimmed : null);
  if (!envelopeJson) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(envelopeJson);
  } catch {
    parsed = parseLooseJsonObject(envelopeJson);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
    return null;

  const wrapper = parsed as Record<string, unknown>;
  const envelope = findValue(wrapper, ENVELOPE_KEY_ALIASES);
  const intent = parseArgumentValue(envelope)
    ? buildIntent(parseArgumentValue(envelope)!)
    : buildIntent(wrapper);
  return toToolCall(intent);
}

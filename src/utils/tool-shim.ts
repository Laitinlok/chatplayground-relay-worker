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

const TOOL_PROMPT_SENTINEL = "[relay-tool-prompt-v1]";
const REASONING_PROMPT_SENTINEL = "[relay-reasoning-prompt-v1]";
const TOOL_NAME_ALIASES: Record<string, string[]> = {
  cron: ["schedule", "scheduler", "reminder", "create_reminder", "cron_add"],
  web_search: ["web.run", "web_search_preview", "browser.search", "search"],
};

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
  if (typeof value === "string") return parseLooseJsonObject(value);
  return null;
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
  _provider?: ToolShimProvider,
): string {
  const forcedName = toolChoiceName(toolChoice);
  const policy =
    toolChoice === "none"
      ? "Do not call a tool. Answer in plain text."
      : toolChoice === "required"
        ? "You must call exactly one listed tool."
        : forcedName
          ? `You must call the tool named "${forcedName}".`
          : "Call a tool only when its result is required. Otherwise answer directly.";

  const catalog = tools
    .map((tool) => {
      const schema = tool.function.parameters ?? { type: "object", properties: {} };
      return [
        `## ${tool.function.name}`,
        `- description: ${tool.function.description ?? ""}`,
        `- parameters: ${JSON.stringify(schema)}`,
      ].join("\n");
    })
    .join("\n");

  return [
    TOOL_PROMPT_SENTINEL,
    "<TOOL_RULES>",
    "You MUST strictly obey the following rules:",
    "1. Each reply is in exactly one state.",
    "   A. Need one or more tools -> output one <TOOL_CALL> block per tool, then stop. More calls are allowed after a <TOOL_RESULT>, including edit/write/replace tools on continue turns.",
    "   B. Task finished -> follow the caller's system instructions and answer in full from every <TOOL_RESULT> above. No tool block.",
    "2. Forbidden:",
    "   - Do not write a <TOOL_RESULT> block. The host returns the real result next turn.",
    "   - Do not invent a tool that is not listed.",
    "   - Do not say you will call a tool, that search is unavailable, or ask the user for a link instead of calling web_search.",
    "3. The block format is exactly:",
    "<TOOL_CALL>",
    "tool: tool_name",
    "params:",
    '{"param":"value"}',
    "</TOOL_CALL>",
    "4. params is one JSON object. Include every required field.",
    policy,
    "</TOOL_RULES>",
    "<AVAILABLE_TOOLS>",
    catalog,
    "</AVAILABLE_TOOLS>",
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
      { ...first, content: `${first.content}\n\n${toolPrompt}` },
      ...rest,
    ];
  }

  return [{ role: "system", content: toolPrompt }, ...messages];
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
}function parseLooseJsonObject(json: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(json);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    return null;
  }
  return null;
}
const XML_TOOL_CALL_RE = /<TOOL_CALL>([\s\S]*?)<\/TOOL_CALL>/i;

/** llm-tool-shim xml_en block: tool name plus one JSON params object. */
function extractJsonObject(source: string, start: number): string | null {
  let depth = 0;
  let end = -1;
  let inString = false;
  let escaped = false;
  for (let i = start; i < source.length; i++) {
    const ch = source[i]!;
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth++;
    else if (ch === "}" && --depth === 0) {
      end = i + 1;
      break;
    }
  }
  return end < 0 ? null : source.slice(start, end);
}

function parseXmlToolCallBody(body: string): ParsedToolIntent | null {
  const name = body.match(/^\s*tool\s*[:=]\s*["']?([^\s"'\n]+)/im)?.[1];
  if (!name) return null;
  const params = body.match(/params\s*[:=]\s*([\s\S]*)/i)?.[1]?.trim() ?? "";
  const jsonStart = params.indexOf("{");
  if (jsonStart < 0) return null;
  const jsonText = extractJsonObject(params, jsonStart);
  if (!jsonText) return null;
  const parsed = parseArgumentValue(jsonText);
  return parsed ? { name, arguments: parsed } : null;
}

function parseXmlToolCalls(text: string): ParsedToolIntent[] {
  return [...text.matchAll(/<TOOL_CALL>([\s\S]*?)<\/TOOL_CALL>/gi)]
    .map((match) => parseXmlToolCallBody(match[1] ?? ""))
    .filter((intent): intent is ParsedToolIntent => intent !== null);
}

export interface ToolCallGateResult {
  calls: ShimToolCall[];
  rejections: string[];
}

function validateToolArguments(
  name: string,
  args: Record<string, unknown>,
  tools?: OpenAITool[],
): string | null {
  const tool = tools?.find((item) => item.function.name === name);
  const schema = tool?.function.parameters;
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return null;
  const required = (schema as { required?: unknown }).required;
  if (!Array.isArray(required)) return null;
  const missing = required.filter(
    (key): key is string =>
      typeof key === "string" && (args[key] === undefined || args[key] === ""),
  );
  if (missing.length === 0) return null;
  return `Tool "${name}" is missing required argument(s): ${missing.join(", ")}.`;
}

function toShimToolCall(
  intent: ParsedToolIntent,
  tools?: OpenAITool[],
): { call: ShimToolCall | null; rejection: string | null } {
  const name = resolveToolName(intent.name, tools);
  if (!name) {
    const listed = tools?.map((tool) => tool.function.name).join(", ") || "(none)";
    return {
      call: null,
      rejection: `Unknown tool "${intent.name}". Use only: ${listed}.`,
    };
  }
  const argumentsValue =
    normalizeName(name) === "web_search"
      ? parseSearchArgumentAliases(intent.arguments)
      : intent.arguments;
  const rejection = validateToolArguments(name, argumentsValue, tools);
  if (rejection) return { call: null, rejection };
  return {
    call: {
      id: `call_${crypto.randomUUID().replace(/-/g, "")}`,
      type: "function",
      function: { name, arguments: JSON.stringify(argumentsValue) },
    },
    rejection: null,
  };
}

/** Inspect incomplete or broken <TOOL_CALL> markup for gate rejections. */
function rejectBrokenToolMarkup(text: string): string[] {
  const rejections: string[] = [];
  const openIndexes: number[] = [];
  const openRe = /<TOOL_CALL\b[^>]*>/gi;
  let openMatch: RegExpExecArray | null;
  while ((openMatch = openRe.exec(text)) !== null) openIndexes.push(openMatch.index);

  for (const start of openIndexes) {
    const afterOpen = text.slice(start);
    const closeIdx = afterOpen.search(/<\/TOOL_CALL>/i);
    const block = closeIdx >= 0 ? afterOpen.slice(0, closeIdx + "</TOOL_CALL>".length) : afterOpen;
    const name = block.match(/\btool\s*[:=]\s*["']?([^\s"'\n<]+)/i)?.[1];
    const params = block.match(/params\s*[:=]\s*([\s\S]*)/i)?.[1] ?? "";
    const jsonStart = params.indexOf("{");
    if (closeIdx < 0) {
      rejections.push(
        name
          ? `Incomplete <TOOL_CALL> for "${name}": missing </TOOL_CALL>. Params must be one valid JSON object with escaped quotes/newlines, then </TOOL_CALL>.`
          : "Incomplete <TOOL_CALL> block: missing </TOOL_CALL>. Close the tag after a valid params JSON object.",
      );
      continue;
    }
    if (jsonStart < 0) {
      rejections.push(
        name
          ? `Tool "${name}" params must be a JSON object after params:.`
          : "TOOL_CALL params must be a JSON object after params:.",
      );
      continue;
    }
    let depth = 0;
    let end = -1;
    let inString = false;
    let escaped = false;
    for (let i = jsonStart; i < params.length; i++) {
      const ch = params[i]!;
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === "{") depth++;
      else if (ch === "}" && --depth === 0) {
        end = i + 1;
        break;
      }
    }
    if (end < 0) {
      rejections.push(
        name
          ? `Tool "${name}" params JSON is truncated or has unescaped quotes/newlines. Emit valid JSON, then </TOOL_CALL>.`
          : "TOOL_CALL params JSON is truncated or invalid. Emit valid JSON, then </TOOL_CALL>.",
      );
      continue;
    }
    const jsonText = params.slice(jsonStart, end);
    if (!parseArgumentValue(jsonText)) {
      rejections.push(
        name
          ? `Tool "${name}" params are not valid JSON. Escape quotes and newlines inside string values.`
          : "TOOL_CALL params are not valid JSON. Escape quotes and newlines inside string values.",
      );
    }
  }
  return rejections;
}

/** Remove tool-call markup so rejected XML is never shown as the answer. */
export function stripToolCallMarkup(text: string): string {
  return text
    .replace(/<TOOL_CALL\b[\s\S]*?<\/TOOL_CALL>/gi, "")
    .replace(/<TOOL_CALL\b[\s\S]*$/gi, "")
    // One-line / whitespace-loose forms: TOOL_CALL tool: ... params: ...
    .replace(/<\/?TOOL_CALL\b[^>]*>/gi, "")
    .replace(/(?:^|\n)\s*tool\s*[:=]\s*\S+[\s\S]*?\bparams\s*[:=][\s\S]*$/gi, "")
    .trim();
}

/** Gate tool calls: keep valid ones, collect rejection reasons for retries. */
export function gateToolCalls(
  text: string,
  tools?: OpenAITool[],
): ToolCallGateResult {
  const trimmed = normalizeToolPayload(text);
  const intents = parseXmlToolCalls(trimmed);
  const rejections = rejectBrokenToolMarkup(trimmed);
  const calls: ShimToolCall[] = [];
  for (const intent of intents) {
    const gated = toShimToolCall(intent, tools);
    if (gated.call) calls.push(gated.call);
    if (gated.rejection) rejections.push(gated.rejection);
  }
  if (
    tools?.length &&
    calls.length === 0 &&
    /<TOOL_CALL\b|\btool\s*[:=]/i.test(trimmed) &&
    !rejections.length
  ) {
    rejections.push(
      "Malformed tool call. Emit a complete <TOOL_CALL> block with tool: and params: as valid JSON, then </TOOL_CALL>.",
    );
  }
  return { calls, rejections };
}

export function tryParseRelayToolCalls(
  text: string,
  tools?: OpenAITool[],
): ShimToolCall[] {
  return gateToolCalls(text, tools).calls;
}

export function tryParseRelayToolCall(
  text: string,
  tools?: OpenAITool[],
): ShimToolCall | null {
  return gateToolCalls(text, tools).calls[0] ?? null;
}

/** User/system note that asks the model to fix rejected tool calls. */
/** Assistant placeholder used when retrying a rejected tool call (never echo raw XML). */
export function rejectedToolCallStub(rejections: string[]): string {
  return `Invalid tool call rejected by gate: ${rejections[0] ?? "malformed tool call"}`;
}

export function toolCallRetrySignal(rejections: string[]): string {
  const reasons = rejections.map((reason, index) => `${index + 1}. ${reason}`).join("\n");
  return [
    "Your previous tool call was rejected by the tool gate.",
    reasons,
    "Retry now with exactly this shape:",
    "<TOOL_CALL>",
    "tool: tool_name",
    "params:",
    '{"param":"value with \\"quotes\\" and \\n newlines escaped"}',
    "</TOOL_CALL>",
    "Rules: one complete block, valid JSON params, escape quotes/newlines inside strings, then close </TOOL_CALL>. Or answer without a tool if none is needed.",
  ].join("\n");
}

import type { ModelEntry } from "../constants/models";
import type { Env } from "../types/env";
import type { ChatCompletionRequest } from "../types/openai";
import {
  buildUpstreamHeaders,
  buildUpstreamRequest,
  endpointUrl,
} from "./upstream-request";
import { collectUpstream } from "./upstream-stream";
import { gateToolCalls, type OpenAITool } from "./tool-shim";

import { needsCurrentWebSearch } from "./search-query";

const SELECTOR_TIMEOUT_MS = 10_000;
const EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5";
const TOOL_MATCH_THRESHOLD = 0.35;
const TOOL_MATCH_MARGIN = 0.025;
const MAX_TOOL_EMBEDDING_TEXT = 2_000;
const MAX_EMBEDDING_TOOLS = 32;

type EmbeddingAI = {
  run: (
    model: string,
    input: { text: string[] },
  ) => Promise<{ data?: number[][] }>;
};

export interface RankedToolCandidates {
  tools: OpenAITool[];
  topScore: number;
  margin: number;
  confident: boolean;
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || a.length !== b.length) return -1;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < a.length; index++) {
    dot += a[index]! * b[index]!;
    normA += a[index]! * a[index]!;
    normB += b[index]! * b[index]!;
  }
  if (normA === 0 || normB === 0) return -1;
  return dot / Math.sqrt(normA * normB);
}

/** Rank advertised tools semantically; callers must still validate the chosen call. */
export async function rankToolsByEmbedding(
  userText: string,
  tools: OpenAITool[],
  ai?: EmbeddingAI,
): Promise<RankedToolCandidates | null> {
  if (!ai || !userText.trim() || tools.length === 0) return null;
  if (tools.length > MAX_EMBEDDING_TOOLS) return null;
  const descriptions = tools.map((tool) =>
    [
      `Tool: ${tool.function.name}`,
      tool.function.description ?? "",
      `Parameters: ${JSON.stringify(tool.function.parameters ?? {})}`,
    ]
      .join("\n")
      .slice(0, MAX_TOOL_EMBEDDING_TEXT),
  );
  try {
    const response = await ai.run(EMBEDDING_MODEL, {
      text: [userText.slice(0, MAX_TOOL_EMBEDDING_TEXT), ...descriptions],
    });
    const vectors = response?.data;
    if (
      !Array.isArray(vectors) ||
      vectors.length !== tools.length + 1 ||
      !vectors.every(
        (vector) =>
          Array.isArray(vector) &&
          vector.length > 0 &&
          vector.every((value) => Number.isFinite(value)),
      )
    )
      return null;
    const ranked = tools
      .map((tool, index) => ({
        tool,
        score: cosineSimilarity(vectors[0]!, vectors[index + 1]!),
        index,
      }))
      .filter((item) => Number.isFinite(item.score) && item.score >= 0)
      .sort((a, b) => b.score - a.score || a.index - b.index);
    if (ranked.length === 0) return null;
    const topScore = ranked[0]!.score;
    const margin = topScore - (ranked[1]?.score ?? 0);
    return {
      tools: ranked.map((item) => item.tool),
      topScore,
      margin,
      confident:
        topScore >= TOOL_MATCH_THRESHOLD && margin >= TOOL_MATCH_MARGIN,
    };
  } catch {
    // Embeddings are an optional ranking signal; unavailable AI must not break routing.
    return null;
  }
}
function parameterNames(tool: OpenAITool): string[] {
  const parameters = tool.function.parameters;
  if (
    !parameters ||
    typeof parameters !== "object" ||
    Array.isArray(parameters)
  )
    return [];
  const properties = (parameters as { properties?: unknown }).properties;
  return properties &&
    typeof properties === "object" &&
    !Array.isArray(properties)
    ? Object.keys(properties)
    : [];
}

function toolNameSuggestsSearch(name: string): boolean {
  const normalized = name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  if (/(?:^|_)(?:conversation|conversations|rag|memory|memories|knowledge|document|vector|semantic)(?:_|$)/i.test(normalized))
    return false;
  return /(?:^|_)(?:web|search|browsing|duckduckgo|ddg)(?:_|$)/i.test(
    normalized,
  );
}

function toolNameSuggestsFetch(name: string): boolean {
  return /(?:web|fetch|scrap|crawl|open|read)/i.test(name);
}

export function isFetchTool(tool: OpenAITool): boolean {
  return (
    toolNameSuggestsFetch(tool.function.name) &&
    parameterNames(tool).includes("url")
  );
}

export function isSearchTool(tool: OpenAITool): boolean {
  const parameters = parameterNames(tool);
  const hasSearchQuery =
    parameters.includes("query") || parameters.includes("search_query");
  const hasWebSearchDefinition = /\b(?:web search|search the web|search the internet|internet search|browser search|duckduckgo)\b/i.test(
    tool.function.description ?? "",
  );
  return (
    hasSearchQuery &&
    (toolNameSuggestsSearch(tool.function.name) || hasWebSearchDefinition)
  );
}

/**
 * Hide only fetch/scrape tools during the initial discovery turn. The relay
 * restores the original tool set after it has stored non-empty search results.
 */
export function prioritizeSearchTools(
  tools: OpenAITool[],
  hasSearchResult = false,
  _userText = "",
): OpenAITool[] {
  if (
    hasSearchResult ||
    (hasExplicitUrl(_userText) &&
      /\b(?:open|read|fetch|visit|summari[sz]e)\b/i.test(_userText)) ||
    !shouldSearchWeb(_userText)
  )
    return tools;
  if (tools.some(isSearchTool)) {
    return tools.filter((tool) => !isFetchTool(tool));
  }
  return tools;
}

export function hasExplicitUrl(value: string): boolean {
  return /https?:\/\/[^\s<>)]+/i.test(value);
}

export function shouldSearchWeb(value: string): boolean {
  if (needsCurrentWebSearch(value) || hasExplicitUrl(value)) return true;
  // "Binary search" is a stable algorithm name, not a request to search the
  // web. Keep the generic search trigger for explicit search instructions.
  const withoutStableAlgorithm = value.replace(/\bbinary\s+search\b/gi, "");
  return /\b(?:search|look\s+up|find\s+online|browse\s+the\s+web|web search|internet|online)\b/i.test(
    withoutStableAlgorithm,
  );
}

function latestUserText(request: ChatCompletionRequest): string {
  const user = [...request.messages]
    .toReversed()
    .find((message) => message.role === "user");
  if (typeof user?.content === "string") return user.content;
  return Array.isArray(user?.content)
    ? user.content
        .filter((part) => part.type === "text")
        .map((part) => part.text)
        .join(" ")
    : "";
}

export function createSearchToolCall(
  userText: string,
  tools: OpenAITool[],
): ReturnType<typeof gateToolCalls>["calls"][number] | null {
  const search = tools.find(isSearchTool);
  if (!search) return null;
  const schema = search.function.parameters as
    | { properties?: Record<string, unknown> }
    | undefined;
  const properties = schema?.properties ?? {};
  const queryKey =
    typeof properties.query === "object"
      ? "query"
      : typeof properties.search_query === "object"
        ? "search_query"
        : null;
  if (!queryKey) return null;
  const markup = `<TOOL_CALL>\ntool: ${search.function.name}\nparams:\n${JSON.stringify({ [queryKey]: userText })}\n</TOOL_CALL>`;
  return gateToolCalls(markup, tools).calls[0] ?? null;
}
/**
 * Runs only for an explicit `tool_choice: "auto"`. It makes tool routing a
 * constrained upstream turn, then validates the selected client tool against
 * the original schemas before the answer-generation turn starts.
 */
export async function selectToolForAutoChoice(
  request: ChatCompletionRequest,
  model: ModelEntry,
  env: Env,
  sessionToken: string,
): Promise<ReturnType<typeof gateToolCalls>["calls"][number] | null> {
  // Omitted tool_choice has OpenAI's `auto` semantics. Route every optional
  // tool turn through the selector so clients such as AnythingLLM cannot let
  // the answer model choose a fetch/scrape tool before discovery search.
  // `required` still permits this relay to select the exact required tool.
  // Only an explicit function name or `none` overrides search-first routing.
  if (
    request.tool_choice !== undefined &&
    request.tool_choice !== "auto" &&
    request.tool_choice !== "required"
  )
    return null;
  if (!request.tools?.length) return null;
  const searchTools = request.tools.filter(isSearchTool);
  const fetchTools = request.tools.filter(isFetchTool);
  const userText = latestUserText(request);
  if (
    hasExplicitUrl(userText) &&
    /\b(?:open|read|fetch|visit|summari[sz]e)\b/i.test(userText)
  )
    return null;
  const wantsWebSearch = shouldSearchWeb(userText);
  const isSelectedFetchTool = (name: string) =>
    fetchTools.some((tool) => tool.function.name === name);
  let candidateTools = request.tools;
  if (!wantsWebSearch) {
    // For non-web requests, use embeddings to shortlist large tool catalogs.
    // Include every original tool in the final schema gate: the shortlist is
    // a ranking hint, never authority to widen permissions or alter validation.
    if (request.tools.length < 2 || request.tools.length > MAX_EMBEDDING_TOOLS)
      return null;
    const ranked = await rankToolsByEmbedding(userText, request.tools, env.AI);
    if (!ranked?.confident) return null;
    candidateTools = ranked.tools.slice(0, 5);
  } else if (searchTools.length === 0 && fetchTools.length === 0) {
    return null;
  }
  const fallbackSearchCall = () =>
    createSearchToolCall(userText, request.tools ?? []);
  const directSearchCall =
    request.tool_choice === undefined ||
    request.tool_choice === "auto" ||
    request.tool_choice === "required"
      ? createSearchToolCall(userText, request.tools)
      : null;
  if (wantsWebSearch && directSearchCall) return directSearchCall;

  // The selector protects search-first routing and uses embedding-shortlisted
  // candidates for other tool requests. It never emits an unvalidated call.

  const selector: ChatCompletionRequest = {
    model: request.model,
    messages: [
      {
        role: "system",
        content: [
          "You select tools; do not answer the user.",
          "Return JSON only.",
          'Return {"action":"call_tool","name":"exact tool name","arguments":{...}} only when a tool is necessary.',
          'Return {"action":"answer"} when no tool is necessary.',
          wantsWebSearch
            ? "If a search tool and a fetch/scrape tool are available, choose the exact listed search tool first. Never substitute web_search, web_fetch, or another invented name."
            : "Choose the most relevant listed tool only when it is necessary to fulfill the request. Otherwise return action=answer.",
          `Search tools: ${JSON.stringify(candidateTools.filter(isSearchTool).map((tool) => tool.function.name))}`,
          `Fetch tools: ${JSON.stringify(candidateTools.filter(isFetchTool).map((tool) => tool.function.name))}`,
          `Available tools: ${JSON.stringify(
            candidateTools.map((tool) => ({
              name: tool.function.name,
              description: tool.function.description,
              parameters: tool.function.parameters,
            })),
          )}`, 
        ].join("\n"),
      },
      { role: "user", content: latestUserText(request) },
    ],
  };
  try {
    const built = buildUpstreamRequest(selector, model);
    const response = await fetch(
      endpointUrl(built.endpoint, env.UPSTREAM_CHAT_URL),
      {
        method: "POST",
        headers: buildUpstreamHeaders(sessionToken, env),
        body: JSON.stringify(built.body),
        signal: AbortSignal.timeout(SELECTOR_TIMEOUT_MS),
      },
    );
    if (!response.ok || !response.body) return null;
    const text = (await collectUpstream(response.body)).content;
    const json = text.match(/\{[\s\S]*\}/)?.[0];
    if (!json) return null;
    const decision = JSON.parse(json) as {
      action?: unknown;
      name?: unknown;
      arguments?: unknown;
    };
    if (
      decision.action !== "call_tool" ||
      typeof decision.name !== "string" ||
      !decision.arguments ||
      typeof decision.arguments !== "object" ||
      Array.isArray(decision.arguments)
    )
      return null;
    if (
      wantsWebSearch &&
      isSelectedFetchTool(decision.name) &&
      !hasSearchResult
    ) {
      // Explicit URLs do not bypass search-first; a fetch is valid only after
      // this relay has stored search results.
      return fallbackSearchCall();
    }
    const markup = `<TOOL_CALL>\ntool: ${decision.name}\nparams:\n${JSON.stringify(decision.arguments)}\n</TOOL_CALL>`;
    return gateToolCalls(markup, request.tools).calls[0] ?? null;
  } catch {
    return null;
  }
}

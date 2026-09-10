import type { Env } from "../types/env";
import type { OpenAIMessage } from "../types/openai";
import type { OpenAITool } from "./tool-shim";

const ROUTE_CACHE_TTL_SECONDS = 24 * 60 * 60;
const MAX_KEYWORDS = 12;
const MIN_KEYWORD_LENGTH = 4;
const STOP_WORDS = new Set([
  "about",
  "after",
  "because",
  "current",
  "from",
  "have",
  "latest",
  "more",
  "please",
  "that",
  "tell",
  "this",
  "what",
  "when",
  "with",
  "would",
]);

function latestUserText(messages: OpenAIMessage[]): string {
  let message: OpenAIMessage | undefined;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") {
      message = messages[i];
      break;
    }
  }
  return typeof message?.content === "string" ? message.content : "";
}

export function toolRouteKeywords(messages: OpenAIMessage[]): string[] {
  const words =
    latestUserText(messages)
      .toLowerCase()
      .match(/[a-z0-9][a-z0-9_-]{3,}/g) ?? [];
  return [...new Set(words)]
    .filter(
      (word) => word.length >= MIN_KEYWORD_LENGTH && !STOP_WORDS.has(word),
    )
    .slice(0, MAX_KEYWORDS);
}

async function digest(value: string): Promise<string> {
  const bytes = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return Array.from(new Uint8Array(bytes), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

function cacheKey(modelId: string, keyword: string): string {
  return `tool-route:${modelId}:${keyword}`;
}

export async function loadCachedToolRoute(
  env: Env,
  modelId: string,
  messages: OpenAIMessage[],
  tools: OpenAITool[],
): Promise<string | null> {
  if (!env.MODEL_CACHE) return null;
  const allowed = new Set(tools.map((tool) => tool.function.name));
  const keywords = toolRouteKeywords(messages);
  if (keywords.length < 2) return null;

  const routes = await Promise.all(
    keywords.map(async (keyword) =>
      env.MODEL_CACHE!.get(cacheKey(modelId, await digest(keyword))),
    ),
  );
  const counts = new Map<string, number>();
  for (const route of routes) {
    if (route && allowed.has(route))
      counts.set(route, (counts.get(route) ?? 0) + 1);
  }
  const ranked: Array<[string, number]> = [];
  for (const entry of counts.entries()) {
    const index = ranked.findIndex((item) => entry[1] > item[1]);
    if (index === -1) ranked.push(entry);
    else ranked.splice(index, 0, entry);
  }
  const best = ranked[0];
  return best && best[1] >= 2 ? best[0] : null;
}

export async function saveCachedToolRoute(
  env: Env,
  modelId: string,
  messages: OpenAIMessage[],
  toolName: string,
): Promise<void> {
  if (!env.MODEL_CACHE || toolRouteKeywords(messages).length < 2) return;
  const keywords = toolRouteKeywords(messages);
  await Promise.all(
    keywords.map(async (keyword) =>
      env.MODEL_CACHE!.put(cacheKey(modelId, await digest(keyword)), toolName, {
        expirationTtl: ROUTE_CACHE_TTL_SECONDS,
      }),
    ),
  );
}

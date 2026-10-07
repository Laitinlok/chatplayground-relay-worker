const SEARCH_TIMEOUT_MS = 3_000;
const MAX_QUERY_CHARS = 300;
const MAX_RESULTS = 50;
const MAX_RESULT_TITLE_CHARS = 180;
const MAX_RESULT_SNIPPET_CHARS = 600;
const MAX_RESULT_CONTEXT_CHARS = 8_000;

export interface CloudflareSearchOptions {
  url?: string;
  token?: string;
  count?: number;
}

export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
}

function searchEndpoint(baseUrl?: string): string | null {
  if (!baseUrl) return null;
  try {
    const base = new URL(baseUrl);
    if (base.protocol !== "https:" && base.protocol !== "http:") return null;
    return new URL("search", base.href.endsWith("/") ? base : `${base.href}/`).toString();
  } catch {
    return null;
  }
}

function compactText(value: string, maxChars: number): string {
  const compacted = value.replace(/\s+/g, " ").trim();
  return compacted.length > maxChars
    ? `${compacted.slice(0, maxChars - 1).trimEnd()}…`
    : compacted;
}

function compactResults(results: WebSearchResult[]): WebSearchResult[] {
  const compacted: WebSearchResult[] = [];
  let totalChars = 0;
  for (const result of results) {
    const item = {
      title: compactText(result.title, MAX_RESULT_TITLE_CHARS),
      url: result.url,
      snippet: compactText(result.snippet, MAX_RESULT_SNIPPET_CHARS),
    };
    const itemChars = item.title.length + item.url.length + item.snippet.length;
    if (compacted.length > 0 && totalChars + itemChars > MAX_RESULT_CONTEXT_CHARS) break;
    compacted.push(item);
    totalChars += itemChars;
  }
  return compacted;
}
function normalizeResult(value: Record<string, unknown>): WebSearchResult | null {
  if (typeof value.url !== "string") return null;
  try {
    const url = new URL(value.url);
    if (url.protocol !== "https:" && url.protocol !== "http:") return null;
    return {
      title: typeof value.title === "string" && value.title.trim() ? value.title : url.toString(),
      url: url.toString(),
      snippet:
        typeof value.description === "string"
          ? value.description
          : typeof value.content === "string"
            ? value.content
            : typeof value.snippet === "string"
              ? value.snippet
              : "",
    };
  } catch {
    return null;
  }
}

const JINA_READER = "https://r.jina.ai/";
const JINA_FETCH_TIMEOUT_MS = 15_000;
const MAX_JINA_FETCH_BYTES = 1_000_000;
const DDG_HTML = "https://html.duckduckgo.com/html/";

function decodeHref(value: string): string {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

/** DuckDuckGo wraps the destination in /l/?uddg=. Return the real URL. */
function destinationUrl(href: string): string | null {
  const decoded = decodeHref(href);
  try {
    const url = new URL(decoded, DDG_HTML);
    const wrapped = url.searchParams.get("uddg");
    const target = wrapped ? new URL(wrapped) : url;
    if (target.protocol !== "https:" && target.protocol !== "http:") return null;
    if (target.hostname.endsWith("duckduckgo.com")) return null;
    return target.toString();
  } catch {
    return null;
  }
}

function parseDuckDuckGoHtml(html: string, count: number): WebSearchResult[] {
  const results: WebSearchResult[] = [];
  const blocks = html.split(/<div[^>]*class="[^"]*result[^"]*"/i).slice(1);
  for (const block of blocks) {
    const link = block.match(/<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i);
    if (!link?.[1]) continue;
    const url = destinationUrl(link[1]);
    if (!url) continue;
    const title = decodeHref(link[2] ?? "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
    const snippet = decodeHref(block.match(/<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>/i)?.[1] ?? "")
      .replace(/<[^>]+>/g, "")
      .replace(/\s+/g, " ")
      .trim();
    results.push({ title: title || url, url, snippet });
    if (results.length >= count) break;
  }
  return compactResults(results);
}

/** Jina Reader returns DuckDuckGo HTML as Markdown, not raw HTML. */
function parseDuckDuckGoMarkdown(markdown: string, count: number): WebSearchResult[] {
  const results: WebSearchResult[] = [];
  const seen = new Set<string>();
  // ## [Title](https://duckduckgo.com/l/?uddg=...)
  const headingRe = /^##\s+\[(.+?)\]\((https?:[^)\s]+)\)/gm;
  let match: RegExpExecArray | null;
  while ((match = headingRe.exec(markdown)) !== null) {
    const title = decodeHref(match[1] ?? "").replace(/\s+/g, " ").trim();
    const url = destinationUrl(match[2] ?? "");
    if (!url || seen.has(url)) continue;
    // Snippet: next non-empty paragraph-like line that is not another heading/image.
    const after = markdown.slice(match.index + match[0].length);
    const snippetLine = after
      .split("\n")
      .map((line) => line.trim())
      .find(
        (line) =>
          line.length > 0 &&
          !line.startsWith("## ") &&
          !line.startsWith("![") &&
          !line.startsWith("[](") &&
          !/^\[(?:Image|![^\]]*)/i.test(line),
      );
    const snippet = decodeHref(snippetLine ?? "")
      .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, "$1")
      .replace(/\*\*/g, "")
      .replace(/\s+/g, " ")
      .trim();
    seen.add(url);
    results.push({ title: title || url, url, snippet });
    if (results.length >= count) break;
  }
  return compactResults(results);
}

function parseCloudflarePayload(payload: unknown, count: number): WebSearchResult[] {
  const rawResults =
    payload && typeof payload === "object" && Array.isArray((payload as { results?: unknown }).results)
      ? ((payload as { results: unknown[] }).results)
      : [];
  return compactResults(
    rawResults
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
      .map(normalizeResult)
      .filter((item): item is WebSearchResult => item !== null)
      .slice(0, count),
  );
}

/** One Jina read of DuckDuckGo. Parses Markdown (Jina) or HTML. */
async function searchDuckDuckGo(query: string, count: number): Promise<WebSearchResult[]> {
  const target = `${DDG_HTML}?q=${encodeURIComponent(query)}`;
  const response = await fetch(`${JINA_READER}${target}`, {
    headers: { accept: "text/plain" },
    signal: AbortSignal.timeout(Math.max(SEARCH_TIMEOUT_MS, 8_000)),
  });
  if (!response.ok) return [];
  const text = await response.text();
  const fromMarkdown = parseDuckDuckGoMarkdown(text, count);
  if (fromMarkdown.length > 0) return fromMarkdown;
  return parseDuckDuckGoHtml(text, count);
}

/** One Cloudflare Search request. Auth via Bearer when a token is configured. */
async function searchCloudflare(
  query: string,
  count: number,
  options: CloudflareSearchOptions,
): Promise<WebSearchResult[]> {
  const endpoint = searchEndpoint(options.url);
  if (!endpoint) return [];
  const headers: Record<string, string> = {
    accept: "application/json",
    "content-type": "application/x-www-form-urlencoded",
  };
  if (options.token) headers.authorization = `Bearer ${options.token}`;
  const form = new URLSearchParams({ q: query });
  if (options.token) form.set("token", options.token);

  // Prefer POST. If the worker rejects it, fall back to GET (live service auths on GET).
  const attempts: Array<() => Promise<Response>> = [
    () =>
      fetch(endpoint, {
        method: "POST",
        headers,
        body: form.toString(),
        signal: AbortSignal.timeout(SEARCH_TIMEOUT_MS),
      }),
    () => {
      const url = new URL(endpoint);
      url.searchParams.set("q", query);
      if (options.token) url.searchParams.set("token", options.token);
      const getHeaders: Record<string, string> = { accept: "application/json" };
      if (options.token) getHeaders.authorization = `Bearer ${options.token}`;
      return fetch(url.toString(), {
        method: "GET",
        headers: getHeaders,
        signal: AbortSignal.timeout(SEARCH_TIMEOUT_MS),
      });
    },
  ];

  for (const run of attempts) {
    try {
      const response = await run();
      if (!response.ok) {
        console.warn("Cloudflare Search returned an error response", {
          query: query.slice(0, 120),
          status: response.status,
        });
        // 401/403 mean auth config is wrong; still try the next transport once.
        if (response.status === 404) continue;
        if (response.status >= 500) continue;
        if (response.status === 401 || response.status === 403) continue;
        return [];
      }
      const payload: unknown = await response.json().catch(() => null);
      const results = parseCloudflarePayload(payload, count);
      if (results.length > 0) return results;
    } catch (error) {
      console.warn("Cloudflare Search request failed", {
        query: query.slice(0, 120),
        error: error instanceof Error ? error.message : "unknown",
      });
    }
  }
  return [];
}

/** Cloudflare Search, then Jina/DuckDuckGo. Empty means the caller falls back to a web_search tool call. */
export async function webSearch(
  query: string,
  options: CloudflareSearchOptions = {},
): Promise<WebSearchResult[]> {
  const normalizedQuery = query.trim().slice(0, MAX_QUERY_CHARS);
  if (!normalizedQuery) return [];
  const count = Number.isInteger(options.count)
    ? Math.min(MAX_RESULTS, Math.max(1, options.count!))
    : 5;
  const sources: Array<[string, () => Promise<WebSearchResult[]>]> = [
    ["Cloudflare Search", () => searchCloudflare(normalizedQuery, count, options)],
    ["DuckDuckGo via Jina", () => searchDuckDuckGo(normalizedQuery, count)],
  ];
  for (const [name, run] of sources) {
    try {
      const results = await run();
      if (results.length > 0) return results;
    } catch (error) {
      console.warn(`${name} failed`, {
        query: normalizedQuery.slice(0, 120),
        error: error instanceof Error ? error.message : "unknown",
      });
    }
  }
  return [];
}

export async function webFetchFromSearchResults(
  url: string,
  results: ReadonlyMap<string, WebSearchResult>,
): Promise<{ title: string; url: string; text: string } | null> {
  let normalizedUrl: string;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return null;
    normalizedUrl = parsed.toString();
  } catch {
    return null;
  }
  const result = results.get(normalizedUrl);
  if (!result) return null;

  try {
    const response = await fetch(`https://r.jina.ai/${normalizedUrl}`, {
      headers: {
        accept: "text/plain",
        "x-respond-with": "text",
      },
      signal: AbortSignal.timeout(JINA_FETCH_TIMEOUT_MS),
    });
    if (response.ok && response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let text = "";
      let bytes = 0;
      try {
        while (bytes < MAX_JINA_FETCH_BYTES) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = value.subarray(0, MAX_JINA_FETCH_BYTES - bytes);
          bytes += chunk.byteLength;
          text += decoder.decode(chunk, { stream: true });
          if (chunk.byteLength < value.byteLength) break;
        }
      } finally {
        await reader.cancel().catch(() => {});
      }
      text += decoder.decode();
      const cleanText = text.trim();
      if (cleanText.length > 0) {
        return { title: result.title, url: result.url, text: cleanText };
      }
    }
  } catch {
    // Fall back to snippet below
  }

  return { title: result.title, url: result.url, text: result.snippet };
}

export function formatWebSearchContext(
  query: string,
  results: readonly WebSearchResult[],
): string {
  if (results.length === 0) {
    return `Web search was requested for: ${query}\\nNo search results were available. Answer cautiously and do not claim that a search was completed.`;
  }
  return [
    "Web search results are available below. Use them as current evidence, cite sources as [N], and do not invent facts not supported by them.",
    `Query: ${query}`,
    ...results.map((result, index) =>
      `[${index + 1}] ${result.title}\\nURL: ${result.url}\\n${result.snippet}`,
    ),
  ].join("\\n\\n");
}

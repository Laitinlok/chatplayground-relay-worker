const SEARCH_TIMEOUT_MS = 3_000;
const MAX_QUERY_CHARS = 300;
const MAX_RESULTS = 50;
const MAX_RESULT_TITLE_CHARS = 180;
const MAX_RESULT_SNIPPET_CHARS = 600;
const MAX_RESULT_CONTEXT_CHARS = 8_000;
const MAX_EXTRACTED_CHARS = 12_000;
const MAX_PASSAGE_CHARS = 2_000;
const MEDIA_URL_RE =
  /(?:https?:\/\/|data:|blob:)[^\s)<>]+\.(?:avif|bmp|gif|ico|jpe?g|png|svg|webp|mp3|mp4|mpeg|ogg|wav|webm)(?:\?[^\s)<>]*)?/gi;
const MEDIA_MARKUP_RE =
  /!\[[^\]]*\]\([^)]*\)|<(?:img|picture|source|video|audio|iframe|canvas)\b[^>]*>[\s\S]*?<\/(?:picture|video|audio|iframe|canvas)>|<(?:img|picture|source|video|audio|iframe|canvas)\b[^>]*\/?>(?:\s*)/gi;

export interface CloudflareSearchOptions {
  url?: string;
  token?: string;
  count?: number;
  youApiKey?: string;
  signal?: AbortSignal;
  provider?: "all" | "configured" | "fallback";
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
    return new URL(
      "search",
      base.href.endsWith("/") ? base : `${base.href}/`,
    ).toString();
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

function extractRelevantText(value: string, query: string): string {
  const cleaned = value
    .replace(MEDIA_MARKUP_RE, " ")
    .replace(MEDIA_URL_RE, " ")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(
      /<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>|<noscript[\s\S]*?<\/noscript>/gi,
      " ",
    )
    .replace(
      /<\/?(?:nav|footer|aside|form|header|menu|button)[^>]*>[\s\S]*?<\/?(?:nav|footer|aside|form|header|menu|button)>/gi,
      " ",
    )
    .replace(/<[^>]+>/g, " ")
    .replace(MEDIA_URL_RE, " ")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/\[[^\]]*\]\((https?:\/\/[^)]+)\)/gi, "")
    .replace(/\r\n?/g, "\n");
  const passages = cleaned
    .split(/\n\s*\n|(?<=[.!?])\s{2,}/)
    .map((part) =>
      part
        .replace(/^[>#*`\-\s]+/, "")
        .replace(/\s+/g, " ")
        .trim(),
    )
    .filter((part) => part.length >= 40 && part.length <= MAX_PASSAGE_CHARS)
    .filter(
      (part) =>
        !/^(accept cookies|cookie policy|privacy policy|subscribe|sign in|advertisement|skip to content)$/i.test(
          part,
        ),
    );
  const unique = [
    ...new Map(passages.map((part) => [part.toLowerCase(), part])).values(),
  ];
  if (unique.length === 0) return compactText(cleaned, MAX_EXTRACTED_CHARS);
  const terms = [...new Set(query.toLowerCase().match(/[a-z0-9]{2,}/g) ?? [])];
  const ranked = unique
    .map((text, index) => {
      const words = text.toLowerCase().match(/[a-z0-9]{2,}/g) ?? [];
      const score = terms.reduce(
        (sum, term) => sum + words.filter((word) => word === term).length,
        0,
      );
      return { text, index, score };
    })
    .toSorted((a, b) => b.score - a.score || a.index - b.index);
  const selected: typeof ranked = [];
  let chars = 0;
  for (const item of ranked) {
    if (chars + item.text.length > MAX_EXTRACTED_CHARS) continue;
    selected.push(item);
    chars += item.text.length + 2;
  }
  selected.sort((a, b) => a.index - b.index);
  return (
    selected.map((item) => item.text).join("\n\n") ||
    compactText(unique[0]!, MAX_EXTRACTED_CHARS)
  );
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
    if (
      compacted.length > 0 &&
      totalChars + itemChars > MAX_RESULT_CONTEXT_CHARS
    )
      break;
    compacted.push(item);
    totalChars += itemChars;
  }
  return compacted;
}
function normalizeResult(
  value: Record<string, unknown>,
): WebSearchResult | null {
  if (typeof value.url !== "string") return null;
  try {
    const url = new URL(value.url);
    if (url.protocol !== "https:" && url.protocol !== "http:") return null;
    return {
      title:
        typeof value.title === "string" && value.title.trim()
          ? value.title
          : url.toString(),
      url: url.toString(),
      snippet:
        typeof value.description === "string"
          ? value.description
          : typeof value.snippet === "string"
            ? value.snippet
            : typeof value.content === "string"
              ? value.content
              : "",
    };
  } catch {
    return null;
  }
}

const JINA_READER = "https://r.jina.ai/";
const JINA_FETCH_TIMEOUT_MS = 8_000;
const MAX_JINA_FETCH_BYTES = 256_000;
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
    if (target.protocol !== "https:" && target.protocol !== "http:")
      return null;
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
    const link = block.match(
      /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i,
    );
    if (!link?.[1]) continue;
    const url = destinationUrl(link[1]);
    if (!url) continue;
    const title = decodeHref(link[2] ?? "")
      .replace(/<[^>]+>/g, "")
      .replace(/\s+/g, " ")
      .trim();
    const snippet = decodeHref(
      block.match(
        /<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>/i,
      )?.[1] ?? "",
    )
      .replace(/<[^>]+>/g, "")
      .replace(/\s+/g, " ")
      .trim();
    results.push({ title: title || url, url, snippet });
    if (results.length >= count) break;
  }
  return compactResults(results);
}

/** Jina Reader returns DuckDuckGo HTML as Markdown, not raw HTML. */
function parseDuckDuckGoMarkdown(
  markdown: string,
  count: number,
): WebSearchResult[] {
  const results: WebSearchResult[] = [];
  const seen = new Set<string>();
  // ## [Title](https://duckduckgo.com/l/?uddg=...)
  const headingRe = /^##\s+\[(.+?)\]\((https?:[^)\s]+)\)/gm;
  let match: RegExpExecArray | null;
  while ((match = headingRe.exec(markdown)) !== null) {
    const title = decodeHref(match[1] ?? "")
      .replace(/\s+/g, " ")
      .trim();
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

function parseCloudflarePayload(
  payload: unknown,
  count: number,
): WebSearchResult[] {
  const rawResults =
    payload &&
    typeof payload === "object" &&
    Array.isArray((payload as { results?: unknown }).results)
      ? (payload as { results: unknown[] }).results
      : [];
  return compactResults(
    rawResults
      .filter((item): item is Record<string, unknown> =>
        Boolean(item && typeof item === "object"),
      )
      .map(normalizeResult)
      .filter((item): item is WebSearchResult => item !== null)
      .slice(0, count),
  );
}

/** One Jina read of DuckDuckGo. Parses Markdown (Jina) or HTML. */
async function searchDuckDuckGo(
  query: string,
  count: number,
  signal: AbortSignal,
): Promise<WebSearchResult[]> {
  const target = `${DDG_HTML}?q=${encodeURIComponent(query)}`;
  const response = await fetch(`${JINA_READER}${target}`, {
    headers: { accept: "text/plain" },
    signal: AbortSignal.any([
      signal,
      AbortSignal.timeout(Math.max(SEARCH_TIMEOUT_MS, 8_000)),
    ]),
  });
  if (!response.ok) return [];
  const text = await response.text();
  const fromMarkdown = parseDuckDuckGoMarkdown(text, count);
  if (fromMarkdown.length > 0) return fromMarkdown;
  return parseDuckDuckGoHtml(text, count);
}

/** One Cloudflare Search request. Auth via Bearer when a token is configured. */
async function searchYouAgents(
  query: string,
  count: number,
  signal: AbortSignal,
): Promise<WebSearchResult[]> {
  const response = await fetch("https://api.you.com/v1/agents/search", {
    method: "POST",
    headers: {
      accept: "application/json",
      "content-type": "application/json",
    },
    body: JSON.stringify({ query, count }),
    signal: AbortSignal.any([signal, AbortSignal.timeout(SEARCH_TIMEOUT_MS)]),
  });
  if (!response.ok) return [];
  const payload = (await response.json().catch(() => null)) as Record<
    string,
    unknown
  > | null;
  const results = payload?.results;
  const raw: unknown[] = Array.isArray(results)
    ? results
    : results &&
        typeof results === "object" &&
        Array.isArray((results as Record<string, unknown>).web)
      ? ((results as Record<string, unknown>).web as unknown[])
      : Array.isArray(payload?.web)
        ? payload.web
        : [];
  return compactResults(
    raw
      .filter((item): item is Record<string, unknown> =>
        Boolean(item && typeof item === "object"),
      )
      .map((item) =>
        normalizeResult({
          title: item.title,
          url: item.url ?? item.link,
          description: Array.isArray(item.snippets)
            ? item.snippets.join(" ")
            : (item.description ?? item.snippet),
        }),
      )
      .filter((item): item is WebSearchResult => item !== null)
      .slice(0, count),
  );
}

async function searchCloudflare(
  query: string,
  count: number,
  options: CloudflareSearchOptions,
  signal: AbortSignal,
): Promise<WebSearchResult[]> {
  const endpoint = searchEndpoint(options.url);
  if (!endpoint) return [];
  const controller = new AbortController();
  const headers: Record<string, string> = {
    accept: "application/json",
    "content-type": "application/x-www-form-urlencoded",
  };
  if (options.token) headers.authorization = `Bearer ${options.token}`;
  const form = new URLSearchParams({ q: query });
  if (options.token) form.set("token", options.token);

  const post = fetch(endpoint, {
    method: "POST",
    headers,
    body: form.toString(),
    signal: AbortSignal.any([
      signal,
      controller.signal,
      AbortSignal.timeout(SEARCH_TIMEOUT_MS),
    ]),
  });
  const getUrl = new URL(endpoint);
  getUrl.searchParams.set("q", query);
  if (options.token) getUrl.searchParams.set("token", options.token);
  const getHeaders: Record<string, string> = { accept: "application/json" };
  if (options.token) getHeaders.authorization = `Bearer ${options.token}`;
  const get = fetch(getUrl.toString(), {
    method: "GET",
    headers: getHeaders,
    signal: AbortSignal.any([
      signal,
      controller.signal,
      AbortSignal.timeout(SEARCH_TIMEOUT_MS),
    ]),
  });

  const parse = async (response: Response): Promise<WebSearchResult[]> => {
    if (!response.ok)
      throw new Error(`Cloudflare Search HTTP ${response.status}`);
    const results = parseCloudflarePayload(
      await response.json().catch(() => null),
      count,
    );
    if (results.length === 0)
      throw new Error("Cloudflare Search returned no results");
    return results;
  };

  try {
    // POST and GET are raced because deployments differ in which transport they
    // accept. The first useful response wins; the slower request is cancelled.
    return await Promise.any([post.then(parse), get.then(parse)]);
  } catch {
    return [];
  } finally {
    controller.abort();
  }
}

/** Race configured search and the DuckDuckGo compatibility fallback. */
export async function webSearch(
  query: string,
  options: CloudflareSearchOptions = {},
): Promise<WebSearchResult[]> {
  const normalizedQuery = query.trim().slice(0, MAX_QUERY_CHARS);
  if (!normalizedQuery) return [];
  const count = Number.isInteger(options.count)
    ? Math.min(MAX_RESULTS, Math.max(1, options.count!))
    : 5;

  const endpoint = searchEndpoint(options.url);
  const controller = new AbortController();
  const signal = options.signal
    ? AbortSignal.any([options.signal, controller.signal])
    : controller.signal;
  const providers = [
    ...(options.provider !== "fallback" && endpoint
      ? [searchCloudflare(normalizedQuery, count, options, signal)]
      : []),
    ...(options.provider !== "configured"
      ? [
          searchDuckDuckGo(normalizedQuery, count, signal),
          searchYouAgents(normalizedQuery, count, signal),
        ]
      : []),
  ];
  const usable = providers.map(async (provider) => {
    const results = await provider;
    if (results.length === 0) throw new Error("empty search result");
    return results;
  });
  try {
    // Each provider has its own timeout. Wait for the first usable result
    // instead of treating a slow provider as a failed search after 500 ms.
    return await Promise.any(usable);
  } catch {
    return [];
  } finally {
    controller.abort();
  }
}

export async function webFetchFromSearchResults(
  url: string,
  results: ReadonlyMap<string, WebSearchResult>,
  query = "",
): Promise<{ title: string; url: string; text: string } | null> {
  let normalizedUrl: string;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:")
      return null;
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
        return {
          title: result.title,
          url: result.url,
          text: extractRelevantText(cleanText, query),
        };
      }
    }
  } catch {
    // Fall back to snippet below
  }

  return {
    title: result.title,
    url: result.url,
    text: extractRelevantText(result.snippet, query),
  };
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
    ...results.map(
      (result, index) =>
        `[${index + 1}] ${result.title}\\nURL: ${result.url}\\n${result.snippet}`,
    ),
  ].join("\\n\\n");
}

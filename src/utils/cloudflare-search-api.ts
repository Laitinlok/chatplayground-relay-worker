const SEARCH_TIMEOUT_MS = 8_000;
const MAX_QUERY_CHARS = 300;
const MAX_RESULTS = 50;

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
            : "",
    };
  } catch {
    return null;
  }
}

export async function webSearch(
  query: string,
  options: CloudflareSearchOptions = {},
): Promise<WebSearchResult[]> {
  const normalizedQuery = query.trim().slice(0, MAX_QUERY_CHARS);
  const endpoint = searchEndpoint(options.url);
  if (!normalizedQuery || !endpoint) {
    console.warn("Cloudflare Search URL or query is missing", {
      hasQuery: Boolean(normalizedQuery),
      hasUrl: Boolean(options.url),
    });
    return [];
  }

  const count = Number.isInteger(options.count)
    ? Math.min(MAX_RESULTS, Math.max(1, options.count!))
    : 5;
  const form = new URLSearchParams({ q: normalizedQuery });
  if (options.token) form.set("token", options.token);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { accept: "application/json", "content-type": "application/x-www-form-urlencoded" },
      body: form.toString(),
      signal: AbortSignal.timeout(SEARCH_TIMEOUT_MS),
    });
    if (!response.ok) {
      console.warn("Cloudflare Search returned HTTP", response.status);
      return [];
    }

    const payload = (await response.json()) as { results?: unknown };
    const rawResults = Array.isArray(payload.results) ? payload.results : [];
    const results = rawResults
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
      .map(normalizeResult)
      .filter((item): item is WebSearchResult => item !== null)
      .slice(0, count);
    console.log("Cloudflare Search", {
      status: response.status,
      count: results.length,
      requestedEngines: "service defaults",
    });
    return results;
  } catch (error) {
    console.warn("Cloudflare Search request failed", String(error));
    return [];
  }
}

/** The configured Cloudflare Search API exposes search results, not extraction. */
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

const SEARCH_TIMEOUT_MS = 8_000;
const MAX_QUERY_CHARS = 300;
const MAX_RESULTS = 50;
const SEARCH_MAX_ATTEMPTS = 3;
const SEARCH_RETRY_DELAY_MS = 250;
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

  for (let attempt = 1; attempt <= SEARCH_MAX_ATTEMPTS; attempt++) {
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { accept: "application/json", "content-type": "application/x-www-form-urlencoded" },
        body: form.toString(),
        signal: AbortSignal.timeout(SEARCH_TIMEOUT_MS),
      });
      const responseText = await response.text();
      let payload: Record<string, unknown> | null = null;
      try {
        const parsed: unknown = JSON.parse(responseText);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          payload = parsed as Record<string, unknown>;
        }
      } catch {
        // Include a bounded body excerpt in diagnostics below for non-JSON errors.
      }

      if (!response.ok) {
        console.warn("Cloudflare Search returned an error response", {
          endpoint,
          query: normalizedQuery.slice(0, 120),
          status: response.status,
          attempt,
          maxAttempts: SEARCH_MAX_ATTEMPTS,
          contentType: response.headers.get("content-type"),
          body: responseText.slice(0, 500),
        });
        // Retry temporary throttling, timeout, and server failures, but do not
        // repeat requests that are rejected for configuration/auth reasons.
        const retryable = response.status === 408 || response.status === 429 || response.status >= 500;
        if (!retryable || attempt === SEARCH_MAX_ATTEMPTS) return [];
      } else {
        const rawResults = Array.isArray(payload?.results) ? payload.results : [];
        const results = compactResults(
          rawResults
            .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
            .map(normalizeResult)
            .filter((item): item is WebSearchResult => item !== null)
            .slice(0, count),
        );
        if (results.length > 0) {
          console.log("Cloudflare Search succeeded", {
            endpoint,
            query: normalizedQuery.slice(0, 120),
            status: response.status,
            count: results.length,
            attempt,
            requestedEngines: "service defaults",
          });
          return results;
        }
        console.warn("Cloudflare Search returned no usable results", {
          endpoint,
          query: normalizedQuery.slice(0, 120),
          status: response.status,
          attempt,
          maxAttempts: SEARCH_MAX_ATTEMPTS,
          responseKeys: payload ? Object.keys(payload) : [],
          reportedResultCount: payload?.number_of_results,
          enabledEngines: payload?.enabled_engines,
          unresponsiveEngines: payload?.unresponsive_engines,
          body: responseText.slice(0, 500),
        });
      }
    } catch (error) {
      console.warn("Cloudflare Search request failed", {
        attempt,
        maxAttempts: SEARCH_MAX_ATTEMPTS,
        error: String(error),
      });
      if (attempt === SEARCH_MAX_ATTEMPTS) return [];
    }

    if (attempt < SEARCH_MAX_ATTEMPTS) {
      await new Promise((resolve) =>
        setTimeout(resolve, SEARCH_RETRY_DELAY_MS * attempt),
      );
    }
  }
  return [];
}

const JINA_FETCH_TIMEOUT_MS = 6_000;
const MAX_JINA_FETCH_BYTES = 64 * 1024;

/**
 * Fetch the full text for a search result using Jina Reader (r.jina.ai).
 * Falls back to the cached snippet if extraction fails or times out.
 */
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

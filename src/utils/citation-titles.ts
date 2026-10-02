import { webSearch } from "./cloudflare-search-api";

const TITLE_FETCH_TIMEOUT_MS = 3000;
const METADATA_FETCH_TIMEOUT_MS = 5000;
const MAX_TITLE_BYTES = 64 * 1024;
const MAX_METADATA_BYTES = 32 * 1024;
const MAX_TITLE_CHARS = 200;

export type CitationTitleMap = ReadonlyMap<string, string>;

export interface CitationTitleOptions {
  searchUrl?: string;
  searchToken?: string;
}


function isBlockedHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/[\[\]]/g, "");
  if (host === "localhost" || host.endsWith(".localhost") || host.endsWith(".local")) return true;
  if (/^(10\.|127\.|169\.254\.|192\.168\.)/.test(host)) return true;
  const private172 = host.match(/^172\.(\d+)\./);
  if (private172 && Number(private172[1]) >= 16 && Number(private172[1]) <= 31) return true;
  return host === "::1" || host.startsWith("fc") || host.startsWith("fd");
}

function unusableTitle(title: string | null | undefined): boolean {
  if (!title) return true;
  return /^(access denied|forbidden|just a moment\.\.\.|attention required|security verification|cloudflare)$/i.test(title.trim());
}

async function readBoundedText(response: Response, maxBytes: number): Promise<string> {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let text = "";
  let bytes = 0;
  try {
    while (bytes < maxBytes) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = value.subarray(0, maxBytes - bytes);
      bytes += chunk.byteLength;
      text += decoder.decode(chunk, { stream: true });
      if (chunk.byteLength < value.byteLength) break;
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
  return text + decoder.decode();
}

async function fetchOgFetchTitle(url: string): Promise<string | null> {
  const endpoint = new URL("https://api.ogfetch.com/preview");
  endpoint.searchParams.set("url", url);
  const response = await fetch(endpoint, {
    headers: { accept: "application/json" },
    signal: AbortSignal.timeout(METADATA_FETCH_TIMEOUT_MS),
  });
  if (!response.ok) return null;
  const text = await readBoundedText(response, MAX_METADATA_BYTES);
  try {
    const payload = JSON.parse(text) as { title?: unknown };
    return normalizeTitle(typeof payload.title === "string" ? payload.title : null);
  } catch {
    return null;
  }
}

async function fetchMicrolinkTitle(url: string): Promise<string | null> {
  const endpoint = new URL("https://api.microlink.io/");
  endpoint.searchParams.set("url", url);
  const response = await fetch(endpoint, {
    headers: { accept: "application/json" },
    signal: AbortSignal.timeout(METADATA_FETCH_TIMEOUT_MS),
  });
  if (!response.ok) return null;
  const text = await readBoundedText(response, MAX_METADATA_BYTES);
  try {
    const payload = JSON.parse(text) as { data?: { title?: unknown } };
    return normalizeTitle(typeof payload.data?.title === "string" ? payload.data.title : null);
  } catch {
    return null;
  }
}

async function fetchJinaTitle(url: string): Promise<string | null> {
  const response = await fetch(`https://r.jina.ai/${url}`, {
    headers: { accept: "text/plain" },
    signal: AbortSignal.timeout(METADATA_FETCH_TIMEOUT_MS),
  });
  if (!response.ok) return null;
  const text = await readBoundedText(response, MAX_METADATA_BYTES);
  const metadataTitle = /^Title:\s*(.+)$/im.exec(text)?.[1];
  const heading = /^#\s+(.+)$/m.exec(text)?.[1];
  return normalizeTitle(metadataTitle ?? heading);
}

function fallbackTitle(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function normalizeTitle(value: string | null | undefined): string | null {
  if (!value) return null;
  const title = value.replace(/\s+/g, " ").trim();
  return title ? title.slice(0, MAX_TITLE_CHARS) : null;
}

async function fetchHtmlTitle(
  response: Response,
): Promise<string | null> {
  if (!response.body) return null;
  let documentTitle = "";
  let openGraphTitle: string | null = null;
  const rewriter = new HTMLRewriter()
    .on("title", {
      text(chunk: { text: string; lastInTextNode: boolean }) {
        documentTitle += chunk.text;
      },
    })
    .on("meta", {
      element(element: { getAttribute(name: string): string | null }) {
        const property = element.getAttribute("property")?.toLowerCase();
        const name = element.getAttribute("name")?.toLowerCase();
        if (property === "og:title" || name === "twitter:title") {
          openGraphTitle = element.getAttribute("content");
        }
      },
    });

  const transformed = rewriter.transform(response);
  const reader = transformed.body?.getReader();
  if (!reader) return null;
  let bytes = 0;
  try {
    while (bytes < MAX_TITLE_BYTES) {
      const { value, done } = await reader.read();
      if (done) break;
      const remaining = MAX_TITLE_BYTES - bytes;
      const chunk = value.subarray(0, remaining);
      bytes += chunk.byteLength;
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
  return normalizeTitle(openGraphTitle) ?? normalizeTitle(documentTitle);
}

export async function resolveCitationTitles(
  urls: readonly string[],
  options: CitationTitleOptions = {},
): Promise<ReadonlyMap<string, string>> {
  const entries = await Promise.all(urls.map(async (url) => {
    const fallback = fallbackTitle(url);
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.toLowerCase().replace(/[\[\]]/g, "");
    if (
      (parsed.protocol !== "https:" && parsed.protocol !== "http:") ||
      parsed.username ||
      parsed.password ||
      isBlockedHost(hostname)
    ) {
      return [url, fallback] as const;
    }
    const response = await fetch(parsed, {
      headers: { accept: "text/html,application/xhtml+xml" },
      redirect: "follow",
      signal: AbortSignal.timeout(TITLE_FETCH_TIMEOUT_MS),
    });
    if (!response.ok) {
      if (options.searchUrl) {
        const searchResults = await webSearch(url, { url: options.searchUrl, token: options.searchToken, count: 1 });
        const searchTitle = searchResults[0]?.title;
        if (!unusableTitle(searchTitle)) return [url, searchTitle!] as const;
      }
      return [url, fallback] as const;
    }
    if (response.url) {
      const finalUrl = new URL(response.url);
      if (
        (finalUrl.protocol !== "https:" && finalUrl.protocol !== "http:") ||
        isBlockedHost(finalUrl.hostname.toLowerCase().replace(/[\\[\\]]/g, ""))
      ) {
        return [url, fallback] as const;
      }
    }
    const title = await fetchHtmlTitle(response);
    if (!unusableTitle(title)) return [url, title!] as const;
    for (const fetcher of [fetchOgFetchTitle, fetchMicrolinkTitle, fetchJinaTitle]) {
      try {
        const alternateTitle = await fetcher(url);
        if (!unusableTitle(alternateTitle)) return [url, alternateTitle!] as const;
      } catch {
        // Continue through the metadata fallback chain.
      }
    }
    if (options.searchUrl) {
      const searchResults = await webSearch(url, {
        url: options.searchUrl,
        token: options.searchToken,
        count: 1,
      });
      const searchTitle = searchResults[0]?.title;
      if (!unusableTitle(searchTitle)) return [url, searchTitle!] as const;
    }
    return [url, fallback] as const;
  } catch {
    return [url, fallback] as const;
  }
  }));
  return new Map(entries);
}

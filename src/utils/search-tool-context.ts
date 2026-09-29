import type { WebSearchResult } from "./cloudflare-search-api";

const MAX_RESULTS = 50;
const MAX_TITLE_CHARS = 180;
const MAX_SNIPPET_CHARS = 600;
const MAX_CONTEXT_CHARS = 8_000;

function compactText(value: string, maxChars: number): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > maxChars
    ? `${normalized.slice(0, maxChars - 1).trimEnd()}…`
    : normalized;
}

export function compactSearchResults(
  results: readonly WebSearchResult[],
): WebSearchResult[] {
  const output: WebSearchResult[] = [];
  let chars = 0;
  for (const result of results.slice(0, MAX_RESULTS)) {
    const compacted = {
      title: compactText(result.title, MAX_TITLE_CHARS),
      url: result.url,
      snippet: compactText(result.snippet, MAX_SNIPPET_CHARS),
    };
    const size = compacted.title.length + compacted.url.length + compacted.snippet.length;
    if (output.length > 0 && chars + size > MAX_CONTEXT_CHARS) break;
    output.push(compacted);
    chars += size;
  }
  return output;
}

/**
 * Compact a client-returned web_search/web_fetch payload without asking the
 * model to summarize it. Unknown tool payloads pass through unchanged.
 */
export function compactSearchToolResult(
  content: string,
  toolName?: string,
): string {
  if (toolName !== "web_search" && toolName !== "web_fetch") return content;
  try {
    const value: unknown = JSON.parse(content);
    if (!value || typeof value !== "object" || Array.isArray(value)) return content;
    const record = value as Record<string, unknown>;
    if (toolName === "web_search" && Array.isArray(record.results)) {
      const results = record.results.filter(
        (item): item is WebSearchResult =>
          Boolean(item && typeof item === "object") &&
          typeof (item as Record<string, unknown>).url === "string" &&
          typeof (item as Record<string, unknown>).title === "string",
      );
      return JSON.stringify({
        ...record,
        results: compactSearchResults(
          results.map((item) => ({
            title: item.title,
            url: item.url,
            snippet:
              typeof item.snippet === "string"
                ? item.snippet
                : "",
          })),
        ),
      });
    }
    if (toolName === "web_fetch") {
      return JSON.stringify({
        ...record,
        title: typeof record.title === "string" ? compactText(record.title, MAX_TITLE_CHARS) : record.title,
        text: typeof record.text === "string" ? compactText(record.text, MAX_CONTEXT_CHARS) : record.text,
      });
    }
    return JSON.stringify(record);
  } catch {
    return content.replace(/\s+/g, " ").trim();
  }
}

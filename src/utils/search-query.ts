const AGORA_MESSAGE_BLOCK_RE = /<\/?ag(?:o|io)ra_[a-z0-9_]+(?:\s[^<>]*)?>/gi;
const GENERIC_MESSAGE_WRAPPER_RE = /<\/?(?:user|assistant|system)_message(?:\s[^<>]*)?>/gi;
const MESSAGE_METADATA_RE = /\b(?:sent_date|sent_at|message_id)\s*=\s*[^\s<>]+/gi;

/** Remove transport wrappers that must not become part of a search query. */
export function sanitizeSearchQuery(value: string): string {
  return value
    .replace(AGORA_MESSAGE_BLOCK_RE, "")
    .replace(GENERIC_MESSAGE_WRAPPER_RE, "")
    .replace(MESSAGE_METADATA_RE, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 300);
}

const CURRENT_EVENT_RE =
  /\b(?:latest|current|today|tonight|yesterday|recent|breaking|this week|right now|news)\b/i;

/** Prompts about current events cannot be answered from model memory. */
export function needsCurrentWebSearch(value: string): boolean {
  return CURRENT_EVENT_RE.test(value);
}

const TOOL_DECLINE_RE =
  /\b(?:unable to|cannot|can't|could not|won't|will not|can not)\b[\s\S]{0,100}\b(?:web search|search the web|run (?:the |a )?search|use (?:the |a )?tool|call (?:the |a )?tool|verify|look up|browse)\b|\b(?:web search|search tool) is (?:unavailable|not available|down)\b|\bshare a link\b|\bfrom this chat\b|\b(?:i(?:'ll| will)|let me)\s+(?:check|look up|search|find|verify|distinguish)\b/i;

/** Model answered that it cannot use a tool instead of emitting the call. */
export function declinedToolUse(value: string): boolean {
  const normalized = value
    .replace(/[\u2018\u2019\u201B\u2032]/g, "'")
    .replace(/[\u201C\u201D]/g, '"');
  return TOOL_DECLINE_RE.test(normalized);
}

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

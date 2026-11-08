export const CHAT_CACHE_TTL_SECONDS = 60 * 60;

export async function chatCacheKey(
  sessionToken: string,
  modelId: string,
  deploymentVersion: string,
  conversationId?: string,
): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(
      `${sessionToken}:${modelId}:${conversationId ?? "default"}`,
    ),
  );
  const sessionHash = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  const hourBucket = Math.floor(Date.now() / (60 * 60 * 1000));
  return `chat:${deploymentVersion}:${hourBucket}:${sessionHash}`;
}

export async function saveCachedChatId(
  cache: KVNamespace | undefined,
  key: string,
  chatId: string | null,
): Promise<void> {
  if (!cache || !chatId) return;
  await cache.put(key, chatId, { expirationTtl: CHAT_CACHE_TTL_SECONDS });
}

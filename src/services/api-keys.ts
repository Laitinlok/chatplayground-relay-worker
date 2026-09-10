// API key format: `rly_<43 url-safe chars>` (~256 bits entropy).
// Only the SHA-256 hash is persisted; the raw key is shown once at creation.
const KEY_PREFIX = "rly_";
const KEY_ENTROPY_BYTES = 32;

export interface ApiKeyRow {
  id: string;
  key_hash: string;
  key_prefix: string;
  label: string;
  created_at: number;
  revoked_at: number | null;
}

export interface CreatedKey {
  id: string;
  key: string;
  key_prefix: string;
  label: string;
  created_at: number;
}

async function sha256Hex(input: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(input),
  );
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export async function createKey(
  db: D1Database,
  label: string,
): Promise<CreatedKey> {
  const bytes = new Uint8Array(KEY_ENTROPY_BYTES);
  crypto.getRandomValues(bytes);
  const raw =
    KEY_PREFIX +
    btoa(String.fromCharCode(...bytes))
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");

  const id = crypto.randomUUID();
  const key_hash = await sha256Hex(raw);
  const key_prefix = raw.slice(0, 12);
  const created_at = Date.now();

  await db
    .prepare(
      "INSERT INTO api_keys (id, key_hash, key_prefix, label, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
    )
    .bind(id, key_hash, key_prefix, label, created_at)
    .run();

  return { id, key: raw, key_prefix, label, created_at };
}

export async function listKeys(db: D1Database): Promise<ApiKeyRow[]> {
  const { results } = await db
    .prepare(
      "SELECT id, key_hash, key_prefix, label, created_at, revoked_at FROM api_keys ORDER BY created_at DESC",
    )
    .all<ApiKeyRow>();
  return results ?? [];
}

export async function revokeKey(db: D1Database, id: string): Promise<boolean> {
  const result = await db
    .prepare(
      "UPDATE api_keys SET revoked_at = ?1 WHERE id = ?2 AND revoked_at IS NULL",
    )
    .bind(Date.now(), id)
    .run();
  return (result.meta.changes ?? 0) > 0;
}

export async function deleteKey(db: D1Database, id: string): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM api_keys WHERE id = ?1")
    .bind(id)
    .run();
  return (result.meta.changes ?? 0) > 0;
}

export async function verifyKey(
  db: D1Database,
  candidate: string,
): Promise<boolean> {
  const hash = await sha256Hex(candidate);
  const row = await db
    .prepare(
      "SELECT id FROM api_keys WHERE key_hash = ?1 AND revoked_at IS NULL LIMIT 1",
    )
    .bind(hash)
    .first<{ id: string }>();
  return row !== null;
}

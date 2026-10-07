import { Hono } from "hono";
import type { Env, Variables } from "../types/env";
import { invalidRequest, upstreamError, OpenAIHTTPError } from "../utils/errors";

const embeddings = new Hono<{ Bindings: Env; Variables: Variables }>();

const TTL_30_DAYS_S = 30 * 24 * 60 * 60;

// Supported models matching cfw-embeddings
const SUPPORTED_MODELS: Record<string, string> = {
  "baai/bge-small-en-v1.5": "@cf/baai/bge-small-en-v1.5",
  "baai/bge-base-en-v1.5": "@cf/baai/bge-base-en-v1.5",
  "baai/bge-large-en-v1.5": "@cf/baai/bge-large-en-v1.5",
  "@cf/baai/bge-small-en-v1.5": "@cf/baai/bge-small-en-v1.5",
  "@cf/baai/bge-base-en-v1.5": "@cf/baai/bge-base-en-v1.5",
  "@cf/baai/bge-large-en-v1.5": "@cf/baai/bge-large-en-v1.5",
  "text-embedding-3-small": "@cf/baai/bge-small-en-v1.5",
  "text-embedding-3-large": "@cf/baai/bge-large-en-v1.5",
};

interface EmbeddingRequestBody {
  input: string | string[];
  text?: string;
  model: string;
  encoding_format?: "float" | "base64";
}

async function sha256Hex(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// OpenAI-compatible /v1/embeddings endpoint with KV caching
embeddings.post("/v1/embeddings", async (c) => {
  let body: EmbeddingRequestBody;
  try {
    body = await c.req.json<EmbeddingRequestBody>();
  } catch {
    throw invalidRequest("Invalid JSON request body.");
  }

  const rawInput = body.input ?? body.text;
  if (!rawInput) {
    throw invalidRequest("Missing required field 'input'.");
  }

  const inputs: string[] = Array.isArray(rawInput)
    ? rawInput
    : typeof rawInput === "string"
      ? [rawInput]
      : [];

  if (inputs.length === 0 || inputs.some((item) => typeof item !== "string")) {
    throw invalidRequest("'input' must be a non-empty string or array of strings.");
  }

  const requestedModel = body.model || "baai/bge-base-en-v1.5";
  const cfModel = SUPPORTED_MODELS[requestedModel];
  if (!cfModel) {
    const available = Object.keys(SUPPORTED_MODELS).filter((m) => !m.startsWith("@cf/")).join(", ");
    throw invalidRequest(`Unsupported model '${requestedModel}'. Available: ${available}`);
  }

  if (!c.env.AI) {
    throw new OpenAIHTTPError(
      503,
      "Cloudflare Workers AI binding (AI) is not configured.",
      "upstream_error",
      "ai_binding_missing",
    );
  }

  const kv = c.env.CHAT_CACHE ?? c.env.MODEL_CACHE;
  const data: Array<{ object: "embedding"; embedding: number[]; index: number }> = [];
  let promptTokens = 0;

  for (let index = 0; index < inputs.length; index++) {
    const text = inputs[index]!;
    promptTokens += Math.ceil(text.length / 4); // token approximation
    const cacheKey = `embed_${await sha256Hex(`${cfModel}:${text}`)}`;

    if (kv) {
      const cached = await kv.get<number[]>(cacheKey, "json");
      if (cached && Array.isArray(cached)) {
        data.push({
          object: "embedding",
          embedding: cached,
          index,
        });
        continue;
      }
    }

    let embedding: number[];
    try {
      const response = await c.env.AI.run(cfModel, { text: [text] }) as {
        data: number[][];
      };
      embedding = response?.data?.[0] ?? [];
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw upstreamError(502, `Workers AI embedding error: ${message}`);
    }

    if (!embedding || embedding.length === 0) {
      throw upstreamError(502, "Workers AI returned an empty embedding.");
    }

    if (kv) {
      await kv.put(cacheKey, JSON.stringify(embedding), {
        expirationTtl: TTL_30_DAYS_S,
      });
    }

    data.push({
      object: "embedding",
      embedding,
      index,
    });
  }

  return c.json({
    object: "list",
    data,
    model: requestedModel,
    usage: {
      prompt_tokens: promptTokens,
      total_tokens: promptTokens,
    },
  });
});

export default embeddings;

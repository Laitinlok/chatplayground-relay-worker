import { Hono } from "hono";
import type { Env, Variables } from "../types/env";
import { invalidRequest, upstreamError } from "../utils/errors";
import { findModel } from "../utils/model-id";
import { getModels } from "../utils/model-discovery";

const images = new Hono<{ Bindings: Env; Variables: Variables }>();

const IMAGE_CACHE_TTL_SECONDS = 60 * 60 * 24; // 24h
const IMAGE_DOWNLOAD_TIMEOUT_MS = 30_000;
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

interface ImageGenerationRequest {
  prompt: string;
  model?: string;
  n?: number;
  size?: string;
  response_format?: "url" | "b64_json";
}

interface CachedImage {
  contentType: string;
  base64: string;
}

function cacheKey(id: string): string {
  return `image:${id}`;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function downloadImage(
  url: string,
): Promise<{ contentType: string; bytes: Uint8Array } | null> {
  try {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(IMAGE_DOWNLOAD_TIMEOUT_MS),
    });
    if (!response.ok || !response.body) return null;
    const contentType =
      response.headers.get("content-type")?.split(";")[0]?.trim() ||
      "image/png";
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;
        total += value.byteLength;
        if (total > MAX_IMAGE_BYTES) return null;
        chunks.push(value);
      }
    } finally {
      await reader.cancel().catch(() => {});
    }
    const bytes = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return { contentType, bytes };
  } catch {
    return null;
  }
}

async function removeUpstreamImage(
  chatUrl: string,
  modelName: string,
  imageId: string,
  sessionToken: string,
  env: Env,
): Promise<void> {
  const endpoint = new URL(
    `../generate-image/${encodeURIComponent(modelName)}/remove/${encodeURIComponent(imageId)}`,
    chatUrl,
  ).toString();
  try {
    await fetch(endpoint, {
      method: "GET",
      headers: {
        authorization: `Bearer ${sessionToken}`,
        origin: env.UPSTREAM_ORIGIN,
        referer: env.UPSTREAM_REFERER,
        accept: "application/json",
      },
      signal: AbortSignal.timeout(15_000),
    });
  } catch {
    // Best-effort history cleanup. The cached/local copy is already safe.
  }
}

/** Serve a previously cached generation. Auth required via /v1/*. */
images.get("/v1/images/cached/:id", async (c) => {
  const id = c.req.param("id");
  if (!id || !/^[A-Za-z0-9_-]+$/.test(id)) {
    throw invalidRequest("Invalid image id.");
  }
  if (!c.env.CHAT_CACHE) {
    throw upstreamError(503, "Image cache is not configured.");
  }
  const raw = await c.env.CHAT_CACHE.get(cacheKey(id));
  if (!raw) throw upstreamError(404, "Cached image not found or expired.");
  let cached: CachedImage;
  try {
    cached = JSON.parse(raw) as CachedImage;
  } catch {
    throw upstreamError(502, "Cached image was corrupted.");
  }
  if (!cached?.base64 || typeof cached.base64 !== "string") {
    throw upstreamError(502, "Cached image was corrupted.");
  }
  return new Response(base64ToBytes(cached.base64), {
    headers: {
      "content-type": cached.contentType || "image/png",
      "cache-control": "private, max-age=86400",
    },
  });
});

// OpenAI Image Generation API mapped to chatplayground's /api/generate-image/{model}
images.post("/v1/images/generations", async (c) => {
  let body: ImageGenerationRequest;
  try {
    body = await c.req.json<ImageGenerationRequest>();
  } catch {
    throw invalidRequest("Invalid JSON request body.");
  }

  if (!body || typeof body.prompt !== "string" || !body.prompt.trim()) {
    throw invalidRequest("Missing or empty required field 'prompt'.");
  }

  const modelId = body.model || "gpt-image-2.5-sunburst";
  const registry = await getModels(c.env);
  const model = findModel(modelId, registry);
  if (model && model.endpoint !== "image") {
    throw invalidRequest(`Model '${modelId}' is not an image generation model.`);
  }

  const upstreamModelName = model ? model.modelName : modelId;
  const chatUrl = c.env.UPSTREAM_CHAT_URL;
  const endpoint = new URL(
    `../generate-image/${encodeURIComponent(upstreamModelName)}`,
    chatUrl,
  ).toString();
  const sessionToken = await c.get("sessionToken")();

  const upstreamBody = {
    prompt: body.prompt,
    apiKey: null,
    model: upstreamModelName,
    size: body.size || "auto",
    noSave: true,
  };

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${sessionToken}`,
      origin: c.env.UPSTREAM_ORIGIN,
      referer: c.env.UPSTREAM_REFERER,
      accept: "application/json",
    },
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(60_000),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw upstreamError(
      res.status,
      `Image generation failed (${res.status}): ${errText.slice(0, 300)}`,
    );
  }

  const result = (await res.json()) as {
    id?: string;
    url?: string;
    prompt?: string;
  };
  if (!result?.url) {
    throw upstreamError(
      502,
      "Upstream image generation succeeded but returned no image URL.",
    );
  }

  const imageId =
    typeof result.id === "string" && result.id.trim()
      ? result.id.trim()
      : crypto.randomUUID().replace(/-/g, "");
  const downloaded = await downloadImage(result.url);
  if (!downloaded) {
    throw upstreamError(502, "Failed to download the generated image.");
  }

  const base64 = bytesToBase64(downloaded.bytes);
  if (c.env.CHAT_CACHE) {
    const payload: CachedImage = {
      contentType: downloaded.contentType,
      base64,
    };
    await c.env.CHAT_CACHE.put(cacheKey(imageId), JSON.stringify(payload), {
      expirationTtl: IMAGE_CACHE_TTL_SECONDS,
    });
  }

  // noSave does not clear image history. Delete upstream after we hold a copy.
  if (result.id) {
    await removeUpstreamImage(
      chatUrl,
      upstreamModelName,
      result.id,
      sessionToken,
      c.env,
    );
  }

  const wantB64 = body.response_format === "b64_json" || !c.env.CHAT_CACHE;
  if (wantB64) {
    return c.json({
      created: Math.floor(Date.now() / 1000),
      data: [
        {
          b64_json: base64,
          revised_prompt: result.prompt ?? body.prompt,
        },
      ],
    });
  }

  const origin = new URL(c.req.url).origin;
  return c.json({
    created: Math.floor(Date.now() / 1000),
    data: [
      {
        url: `${origin}/v1/images/cached/${encodeURIComponent(imageId)}`,
        revised_prompt: result.prompt ?? body.prompt,
      },
    ],
  });
});

export default images;

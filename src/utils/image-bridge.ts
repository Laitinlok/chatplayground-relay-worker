import type { Env } from "../types/env";
import type { OpenAIMessage, OpenAIContentPart } from "../types/openai";
import { UPLOAD_TIMEOUT } from "../constants/timeouts";
import { upstreamError } from "./errors";

const MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024;

function decodeDataUrl(
  value: string,
): { bytes: Uint8Array; contentType: string } | null {
  const match = value.match(
    /^data:(image\/[a-z0-9.+-]+);base64,([a-z0-9+/=\s]+)$/i,
  );
  if (!match) return null;
  const binary = atob(match[2]!.replace(/\s+/g, ""));
  if (binary.length > MAX_INLINE_IMAGE_BYTES) {
    throw upstreamError(413, "Inline image exceeds the 20 MB limit.");
  }
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return { bytes, contentType: match[1]! };
}

/** Upload inline image inputs once and replace them with upstream URLs. */
export async function bridgeInlineImages(
  messages: OpenAIMessage[],
  env: Env,
  sessionToken: string,
): Promise<void> {
  for (const message of messages) {
    if (!Array.isArray(message.content)) continue;
    for (const part of message.content as OpenAIContentPart[]) {
      if (part.type !== "image_url") continue;
      const decoded = decodeDataUrl(part.image_url.url);
      if (!decoded) continue;
      const form = new FormData();
      form.append(
        "file",
        new File([decoded.bytes], "image-input", { type: decoded.contentType }),
      );
      const response = await fetch(env.UPSTREAM_UPLOAD_URL, {
        method: "POST",
        headers: {
          authorization: `Bearer ${sessionToken}`,
          origin: env.UPSTREAM_ORIGIN,
          referer: env.UPSTREAM_REFERER,
          accept: "*/*",
        },
        body: form,
        signal: AbortSignal.timeout(UPLOAD_TIMEOUT),
      });
      if (!response.ok) {
        throw upstreamError(
          response.status,
          "Failed to upload inline image input.",
        );
      }
      const body = (await response.json().catch(() => null)) as {
        url?: string;
      } | null;
      if (!body?.url) throw upstreamError(502, "Image upload returned no URL.");
      part.image_url.url = body.url;
    }
  }
}

export function resetImageBridgeForTests(): void {
  // Deliberately empty: the bridge is request-scoped and has no cache.
}

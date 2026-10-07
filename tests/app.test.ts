import { Hono } from "hono";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import app from "../src/index";
import { errorHandler } from "../src/middleware/error-handler";
import { upstreamError } from "../src/utils/errors";
import { resetModelCache } from "../src/utils/model-discovery";

// The wiring in index.ts: what is public, what auth guards, and what CORS
// advertises. Routed through the real app, not a rebuilt one.

const env = {
  UPSTREAM_CHAT_URL: "https://up.example.test/api/chat/azure",
  UPSTREAM_ORIGIN: "https://web.example.test",
  UPSTREAM_REFERER: "https://web.example.test/",
  UPSTREAM_UPLOAD_URL: "https://up.example.test/api/upload",
  CLERK_FAPI_URL: "https://clerk.example.test",
};

beforeEach(() => {
  resetModelCache();
  vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json([]));
});
afterEach(() => vi.restoreAllMocks());

describe("app wiring", () => {
  it("serves the index without auth and without touching upstream", async () => {
    const res = await app.request("/", {}, env);
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({
      name: "chatplayground",
      endpoints: [
        "/v1/models",
        "/v1/models/capabilities",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/files",
        "/v1/images/generations",
        "/v1/embeddings",
      ],
    });
    expect(vi.mocked(fetch)).not.toHaveBeenCalled();
  });

  it("guards every /v1/* route behind auth", async () => {
    for (const path of ["/v1/models", "/v1/chat/completions", "/v1/files"]) {
      const res = await app.request(path, { method: "POST" }, env);
      expect(res.status, path).toBe(401);
    }
    expect(vi.mocked(fetch)).not.toHaveBeenCalled();
  });

  it("answers a CORS preflight with the headers SDKs send", async () => {
    const res = await app.request(
      "/v1/chat/completions",
      {
        method: "OPTIONS",
        headers: {
          origin: "https://client.example.test",
          "access-control-request-method": "POST",
          "access-control-request-headers": "authorization",
        },
      },
      env,
    );
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
    const allowed = res.headers.get("access-control-allow-headers") ?? "";
    expect(allowed.toLowerCase()).toContain("authorization");
  });

  it("404s an unknown path", async () => {
    const res = await app.request("/nope", {}, env);
    expect(res.status).toBe(404);
  });
});

describe("errorHandler fallback", () => {
  it("marks upstream rate limits as non-retryable", async () => {
    const limited = new Hono<any>();
    limited.onError(errorHandler);
    limited.get("/", () => {
      throw upstreamError(429, "too many requests");
    });

    const res = await limited.request("/");
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("60");
    expect(res.headers.get("x-should-retry")).toBe("false");
  });
  it("wraps a non-OpenAIHTTPError as a 500 in the OpenAI envelope", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    // oxlint-disable-next-line typescript/no-explicit-any -- minimal test env stub
    const boom = new Hono<any>();
    boom.onError(errorHandler);
    boom.get("/", () => {
      throw new Error("something unplanned");
    });

    const res = await boom.request("/");
    expect(res.status).toBe(500);
    expect(await res.json()).toEqual({
      error: {
        message: "something unplanned",
        type: "internal_error",
        code: null,
        param: null,
      },
    });
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import app from "../src/index";

const JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyXzEyMyJ9.c2ln-bmF0dXJl";
const PNG_BYTES = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 0]);

function fakeCache() {
  const store = new Map<string, string>();
  return {
    store,
    get: async (key: string) => store.get(key) ?? null,
    put: async (key: string, value: string) => {
      store.set(key, value);
    },
    delete: async (key: string) => {
      store.delete(key);
    },
  };
}

const baseEnv = {
  UPSTREAM_CHAT_URL: "https://app.chatplayground.ai/api/chat/azure",
  UPSTREAM_ORIGIN: "https://web.chatplayground.ai",
  UPSTREAM_REFERER: "https://web.chatplayground.ai/",
};

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("POST /v1/images/generations", () => {
  it("rejects unauthenticated requests with 401", async () => {
    const res = await app.request(
      "/v1/images/generations",
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: "A cat" }),
      },
      baseEnv,
    );
    expect(res.status).toBe(401);
  });

  it("caches the image, removes upstream history, and returns a relay URL", async () => {
    const cache = fakeCache();
    const env = { ...baseEnv, CHAT_CACHE: cache };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/api/models")) {
        return Response.json([
          {
            botId: "gpt-image-2.5-sunburst",
            modelName: "gpt-image-2.5-sunburst",
            provider: "OpenAI",
            group: "image",
            endpoint: "image",
            premiumOnly: false,
          },
        ]);
      }
      if (url.endsWith("/api/generate-image/gpt-image-2.5-sunburst")) {
        const body = JSON.parse(String(init?.body));
        expect(body.size).toBe("1024x1024");
        expect(body.noSave).toBe(true);
        return Response.json({
          id: "img_123",
          url: "https://cdn.chatplayground.ai/image-generation/img_123.png",
          prompt: "A red sports car",
        });
      }
      if (url === "https://cdn.chatplayground.ai/image-generation/img_123.png") {
        return new Response(PNG_BYTES, {
          headers: { "content-type": "image/png" },
        });
      }
      if (url.includes("/generate-image/gpt-image-2.5-sunburst/remove/img_123")) {
        expect(init?.method ?? "GET").toBe("GET");
        return Response.json({ ok: true });
      }
      return new Response("Not found", { status: 404 });
    });

    const res = await app.request(
      "/v1/images/generations",
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${JWT}`,
        },
        body: JSON.stringify({
          prompt: "A red sports car",
          model: "gpt-image-2.5-sunburst",
          size: "1024x1024",
        }),
      },
      env,
    );
    expect(res.status).toBe(200);
    const payload = await res.json() as { data: Array<{ url?: string; b64_json?: string; revised_prompt?: string }> };
    expect(payload.data[0]!.url).toContain("/v1/images/cached/img_123");
    expect(payload.data[0]!.revised_prompt).toBe("A red sports car");
    expect(cache.store.has("image:img_123")).toBe(true);
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes("/generate-image/gpt-image-2.5-sunburst/remove/img_123"),
      ),
    ).toBe(true);

    const cached = await app.request(
      "/v1/images/cached/img_123",
      { headers: { authorization: `Bearer ${JWT}` } },
      env,
    );
    expect(cached.status).toBe(200);
    expect(cached.headers.get("content-type")).toBe("image/png");
    const bytes = new Uint8Array(await cached.arrayBuffer());
    expect([...bytes.slice(0, 8)]).toEqual([...PNG_BYTES.slice(0, 8)]);
  });

  it("returns b64_json when requested and still removes upstream history", async () => {
    const cache = fakeCache();
    const env = { ...baseEnv, CHAT_CACHE: cache };
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/api/models")) {
        return Response.json([
          {
            botId: "gpt-image-2.5-sunburst",
            modelName: "gpt-image-2.5-sunburst",
            provider: "OpenAI",
            group: "image",
            endpoint: "image",
            premiumOnly: false,
          },
        ]);
      }
      if (url.endsWith("/api/generate-image/gpt-image-2.5-sunburst")) {
        return Response.json({
          id: "img_b64",
          url: "https://cdn.chatplayground.ai/image-generation/img_b64.png",
        });
      }
      if (url === "https://cdn.chatplayground.ai/image-generation/img_b64.png") {
        return new Response(PNG_BYTES, {
          headers: { "content-type": "image/png" },
        });
      }
      if (url.includes("/remove/img_b64")) return Response.json({ ok: true });
      return new Response("Not found", { status: 404 });
    });

    const res = await app.request(
      "/v1/images/generations",
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${JWT}`,
        },
        body: JSON.stringify({
          prompt: "A cat",
          response_format: "b64_json",
        }),
      },
      env,
    );
    expect(res.status).toBe(200);
    const payload = await res.json() as { data: Array<{ url?: string; b64_json?: string; revised_prompt?: string }> };
    expect(payload.data[0]!.b64_json).toBeTruthy();
    expect(payload.data[0]!.url).toBeUndefined();
  });

  it("defaults size to auto when omitted", async () => {
    const cache = fakeCache();
    const env = { ...baseEnv, CHAT_CACHE: cache };
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/api/models")) {
        return Response.json([
          {
            botId: "gpt-image-2.5-sunburst",
            modelName: "gpt-image-2.5-sunburst",
            provider: "OpenAI",
            group: "image",
            endpoint: "image",
            premiumOnly: false,
          },
        ]);
      }
      if (url.endsWith("/api/generate-image/gpt-image-2.5-sunburst")) {
        const body = JSON.parse(String(init?.body));
        expect(body.size).toBe("auto");
        return Response.json({
          id: "img_auto",
          url: "https://cdn.chatplayground.ai/image-generation/auto.png",
        });
      }
      if (url.includes("auto.png")) {
        return new Response(PNG_BYTES, { headers: { "content-type": "image/png" } });
      }
      if (url.includes("/remove/img_auto")) return Response.json({ ok: true });
      return new Response("Not found", { status: 404 });
    });

    const res = await app.request(
      "/v1/images/generations",
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${JWT}`,
        },
        body: JSON.stringify({ prompt: "A cat" }),
      },
      env,
    );
    expect(res.status).toBe(200);
  });
});

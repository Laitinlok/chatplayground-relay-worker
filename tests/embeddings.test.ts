import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import app from "../src/index";

const JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyXzEyMyJ9.c2ln-bmF0dXJl";

const mockKV = () => {
  const store = new Map<string, string>();
  return {
    get: vi.fn(async (key: string, type?: string) => {
      const val = store.get(key);
      if (!val) return null;
      return type === "json" ? JSON.parse(val) : val;
    }),
    put: vi.fn(async (key: string, val: string) => {
      store.set(key, val);
    }),
  } as unknown as KVNamespace;
};

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("POST /v1/embeddings", () => {
  it("rejects unauthenticated requests with 401", async () => {
    const res = await app.request("/v1/embeddings", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input: "hello" }),
    }, {});
    expect(res.status).toBe(401);
  });

  it("rejects missing input with 400", async () => {
    const res = await app.request("/v1/embeddings", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${JWT}`,
      },
      body: JSON.stringify({ model: "baai/bge-base-en-v1.5" }),
    }, {});
    expect(res.status).toBe(400);
  });

  it("returns 503 when Workers AI binding is missing", async () => {
    const res = await app.request("/v1/embeddings", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${JWT}`,
      },
      body: JSON.stringify({
        input: "hello world",
        model: "baai/bge-base-en-v1.5",
      }),
    }, {});
    expect(res.status).toBe(503);
  });

  it("computes embeddings and uses KV caching (cfw-embeddings architecture)", async () => {
    const runMock = vi.fn().mockResolvedValue({
      data: [[0.1, 0.2, 0.3]],
    });
    const kv = mockKV();
    const env = {
      AI: { run: runMock },
      CHAT_CACHE: kv,
    };

    // 1. First call: miss cache, calls AI.run
    const res1 = await app.request("/v1/embeddings", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${JWT}`,
      },
      body: JSON.stringify({
        input: ["hello", "world"],
        model: "baai/bge-base-en-v1.5",
      }),
    }, env);

    expect(res1.status).toBe(200);
    const json1 = await res1.json() as any;
    expect(json1.object).toBe("list");
    expect(json1.data).toHaveLength(2);
    expect(json1.data[0].embedding).toEqual([0.1, 0.2, 0.3]);
    expect(runMock).toHaveBeenCalledTimes(2);
    expect(kv.put).toHaveBeenCalledTimes(2);

    // 2. Second call: hits cache
    runMock.mockClear();
    const res2 = await app.request("/v1/embeddings", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${JWT}`,
      },
      body: JSON.stringify({
        input: "hello",
        model: "baai/bge-base-en-v1.5",
      }),
    }, env);

    expect(res2.status).toBe(200);
    const json2 = await res2.json() as any;
    expect(json2.data[0].embedding).toEqual([0.1, 0.2, 0.3]);
    expect(runMock).not.toHaveBeenCalled();
  });
});

import { Hono } from "hono";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { errorHandler } from "../src/middleware/error-handler";
import models from "../src/routes/models";
import type { ModelList } from "../src/types/openai";
import { resetModelCache } from "../src/utils/model-discovery";

const env = { UPSTREAM_CHAT_URL: "https://up.example.test/api/chat/azure" };

const FEED = [
  {
    botId: "gpt-5.6-luna",
    modelName: "gpt-5.6-luna",
    provider: "OpenAI",
    group: "chat",
    endpoint: "azure",
    premiumOnly: false,
  },
  {
    botId: "grok-4.6",
    modelName: "grok-4.6",
    provider: "xAI",
    group: "chat",
    endpoint: "lmsys",
    premiumOnly: true,
  },
  // Image models are returned on the second OpenAI cursor page for Agora.
  {
    botId: "gpt-image-2.5-sunburst",
    modelName: "gpt-image-2.5-sunburst",
    provider: "OpenAI",
    group: "image",
    endpoint: "image",
    premiumOnly: false,
  },
  // Audio model — discovery drops it before the listing ever sees it.
  {
    botId: "tts-bot",
    modelName: "tts-bot",
    provider: "OpenAI",
    group: "audio",
  },
];

// oxlint-disable-next-line typescript/no-explicit-any -- minimal test env stub
function get(path = "/v1/models", testEnv: any = env) {
  // oxlint-disable-next-line typescript/no-explicit-any -- minimal test env stub
  const app = new Hono<any>();
  app.onError(errorHandler);
  app.route("/", models);
  return app.request(path, {}, testEnv);
}

beforeEach(() => {
  resetModelCache();
  vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json(FEED));
});
afterEach(() => vi.restoreAllMocks());

describe("GET /v1/models", () => {
  it("lists non-premium chat models in the OpenAI shape", async () => {
    const res = await get();
    expect(res.status).toBe(200);
    const body = (await res.json()) as ModelList;

    expect(body.object).toBe("list");
    expect(body.data).toEqual([
      {
        id: "gpt-5.6-luna",
        object: "model",
        created: expect.any(Number),
        owned_by: "openai",
      },
    ]);
  });

  it("returns image generation models on the cursor page", async () => {
    const firstPage = (await (await get()).json()) as ModelList;
    expect(firstPage).toMatchObject({
      has_more: true,
      last_id: "gpt-5.6-luna",
    });

    const secondPage = (await (
      await get(`/v1/models?after=${firstPage.last_id}`)
    ).json()) as ModelList;
    expect(secondPage).toEqual({
      object: "list",
      data: [
        {
          id: "gpt-image-2.5-sunburst",
          object: "model",
          created: expect.any(Number),
          owned_by: "openai",
        },
      ],
    });
  });

  it("returns an empty final page for an unknown cursor", async () => {
    const body = (await (
      await get("/v1/models?after=unknown")
    ).json()) as ModelList;
    expect(body).toEqual({ object: "list", data: [] });
  });

  it("hides premiumOnly models unless PREMIUM_MODELS is set", async () => {
    const res = await get("/v1/models", { ...env, PREMIUM_MODELS: "true" });
    const ids = ((await res.json()) as ModelList).data.map((m) => m.id);
    expect(ids).toEqual(["gpt-5.6-luna", "grok-4.6"]);
  });

  it("exposes image models through capabilities for image-generation selectors", async () => {
    const app = new Hono<any>();
    app.onError(errorHandler);
    app.route("/", models);
    const res = await app.request("/v1/models/capabilities", {}, env);
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;
    expect(body.data).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "gpt-image-2.5-sunburst",
          capabilities: ["image"],
          modalities: { input: ["text"], output: ["image"] },
        }),
      ]),
    );
  });

  it("503s when discovery fails instead of serving an empty list", async () => {
    vi.mocked(fetch).mockResolvedValue(new Response("", { status: 500 }));
    vi.spyOn(console, "error").mockImplementation(() => {});

    const res = await get();
    expect(res.status).toBe(503);
    const body = (await res.json()) as { error: { code: string } };
    expect(body.error.code).toBe("model_discovery_failed");
  });
});

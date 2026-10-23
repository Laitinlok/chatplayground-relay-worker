import { Hono } from "hono";
import type { Env, Variables } from "../types/env";
import type { ModelList } from "../types/openai";
import { getModels, isVisible } from "../utils/model-discovery";

const models = new Hono<{ Bindings: Env; Variables: Variables }>();

models.get("/v1/models", async (c) => {
  const registry = await getModels(c.env);
  const created = Math.floor(Date.now() / 1000);
  const visible = registry.filter((m) => isVisible(m, c.env));
  const chatModels = visible.filter((m) => m.endpoint !== "image");
  const imageModels = visible.filter((m) => m.endpoint === "image");
  const after = c.req.query("after");
  const lastChatModelId = chatModels.at(-1)?.id;

  // Agora follows OpenAI cursor pagination when `has_more` is set. This keeps
  // image-generation models off the initial catalog used by clients that only
  // consume the first page, while allowing Agora to synchronize both pages.
  const page =
    after === lastChatModelId ? imageModels : after ? [] : chatModels;
  const hasMore = !after && chatModels.length > 0 && imageModels.length > 0;
  const list: ModelList = {
    object: "list",
    data: page.map((m) => ({
      id: m.id,
      object: "model",
      created,
      owned_by: m.provider,
    })),
    ...(hasMore ? { has_more: true, last_id: lastChatModelId } : {}),
  };
  return c.json(list);
});

// LocalAI-compatible /v1/models/capabilities endpoint for AnythingLLM
// and other clients to detect image generation and other model capabilities.
const handleCapabilities = async (c: any) => {
  const registry = await getModels(c.env);
  const created = Math.floor(Date.now() / 1000);
  const visible = registry.filter((m) => isVisible(m, c.env));

  const data = visible.map((m) => {
    const isImage = m.endpoint === "image";
    return {
      id: m.id,
      object: "model",
      created,
      owned_by: m.provider,
      // AnythingLLM checks: model?.capabilities?.includes("image")
      // LocalAI API documentation also mentions boolean flags/strings.
      // We support both array format and object format for maximum compatibility.
      capabilities: isImage ? ["image"] : ["chat", "completion"],
      modalities: {
        input: isImage ? ["text"] : ["text", "image"],
        output: isImage ? ["image"] : ["text"],
      },
    };
  });

  return c.json({
    object: "list",
    data,
  });
};

models.get("/v1/models/capabilities", handleCapabilities);
// AnythingLLM calls `${url}/models/capabilities` (if user enters base URL with /v1, it calls /v1/models/capabilities; if without /v1, /models/capabilities)
models.get("/models/capabilities", handleCapabilities);

export default models;

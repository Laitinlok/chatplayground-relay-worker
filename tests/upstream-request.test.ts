import { describe, expect, it } from "vitest";
import { toEndpoint } from "../src/constants/endpoints";
import type { ModelEntry } from "../src/constants/models";
import type { ChatCompletionRequest } from "../src/types/openai";
import {
  buildUpstreamHeaders,
  buildUpstreamRequest,
  endpointUrl,
  requiresNewPromptOnly,
} from "../src/utils/upstream-request";

const AZURE_MODEL: ModelEntry = {
  id: "gpt-5.5",
  modelName: "gpt-5.5",
  upstreamModel: "openai/gpt-5.5",
  upstreamBotId: "gpt-5.5",
  provider: "openai",
  endpoint: "azure",
};

const LUNA_MODEL: ModelEntry = {
  id: "gpt-6-luna",
  modelName: "gpt-6-luna",
  upstreamModel: "openai/gpt-6-luna",
  upstreamBotId: "gpt-6-luna",
  provider: "openai",
  endpoint: "azure",
};

const AGENT_MODEL: ModelEntry = {
  id: "gpt-agent",
  modelName: "gpt-agent",
  upstreamModel: "openai/gpt-agent",
  upstreamBotId: "gpt-agent",
  provider: "openai",
  endpoint: "azure",
};

const PERPLEXITY_MODEL: ModelEntry = {
  id: "perplexity-sonar-pro",
  modelName: "sonar-pro",
  upstreamModel: "perplexity/sonar-pro",
  upstreamBotId: "perplexity-sonar-pro",
  provider: "perplexity",
  endpoint: "perplexity",
};

// lmsys modelName is a full provider slug (with a slash) in the live feed.
const LMSYS_MODEL: ModelEntry = {
  id: "llama-4-scout",
  modelName: "meta-llama/llama-4-scout-17b-16e-instruct",
  upstreamModel: "meta/meta-llama/llama-4-scout-17b-16e-instruct",
  upstreamBotId: "llama-4-scout",
  provider: "meta",
  endpoint: "lmsys",
};

const AZURE_BASE_URL = "https://app.chatplayground.ai/api/chat/azure";

describe("buildUpstreamRequest — field mapping", () => {
  it("sends only the latest user prompt to agent models", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-agent",
        user: "client-chat-id",
        messages: [
          { role: "system", content: "System instructions stay client-side." },
          { role: "user", content: "Previous question" },
          { role: "assistant", content: "Previous answer" },
          { role: "user", content: "Current prompt" },
        ],
      },
      AGENT_MODEL,
    );

    expect(body.messages).toEqual([{ role: "user", content: "Current prompt" }]);
    expect(body.chatId).toBe("client-chat-id");
    expect(body.noSave).toBe(true);
  });

  it("sends only the latest user prompt and keeps Luna noSave enabled", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-6-luna",
        messages: [
          { role: "user", content: "hello" },
          { role: "assistant", content: "Hi!" },
          { role: "user", content: "what is an API" },
        ],
      },
      LUNA_MODEL,
    );

    expect(body.messages).toEqual([
      {
        role: "user",
        content: "hello\n\nDraft so far (do not repeat it):\nHi!",
      },
      { role: "user", content: "what is an API" },
    ]);
    expect(body.chatId).toBe("");
    expect(body.noSave).toBe(true);
    expect(body.submissionId).toMatch(/^[0-9a-f-]{36}$/i);
  });

  it("sends only the new user prompt when resuming an upstream chat", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-5.5",
        user: "cmvjv8i7d002pl708e4uhhyyh",
        messages: [
          { role: "user", content: "hello" },
          { role: "assistant", content: "Hi!" },
          { role: "user", content: "what is an API" },
        ],
      },
      AZURE_MODEL,
    );

    expect(body.messages).toEqual([{ role: "user", content: "what is an API" }]);
    expect(body.chatId).toBe("cmvjv8i7d002pl708e4uhhyyh");
  });

  it("defaults to noSave=true and empty chatId", () => {
    const { body } = buildUpstreamRequest(
      { model: "gpt-5.5", messages: [{ role: "user", content: "hi" }] },
      AZURE_MODEL,
    );
    expect(body.noSave).toBe(true);
    expect(body.chatId).toBe("");
    expect(body.botId).toBe("gpt-5.5");
  });

  it("splits messages at the upstream 15,000-character limit", () => {
    const content = "a".repeat(15_001);
    const { body } = buildUpstreamRequest(
      { model: "gpt-5.5", messages: [{ role: "user", content }] },
      AZURE_MODEL,
    );
    expect(body.messages).toHaveLength(2);
    expect(body.messages[0]?.content).toHaveLength(15_000);
    expect(body.messages[1]?.content).toBe("a");
    expect(body.messages.map((message) => message.content).join("")).toBe(
      content,
    );
  });

  it("maps developer messages to system and safely preserves null content", () => {
 const { body } = buildUpstreamRequest(
 {
 model: "gpt-5.5",
 messages: [
 { role: "developer", content: "Follow these rules." },
 { role: "assistant", content: null },
 { role: "user", content: "hi" },
 ],
 },
 AZURE_MODEL,
 );
 expect(body.messages.slice(0, 2)).toEqual([
 { role: "system", content: "Follow these rules." },
 { role: "assistant", content: "" },
 ]);
 });

 it("does not split a surrogate pair at the upstream message limit", () => {
 const content = "a".repeat(14_999) + "😀" + "b";
 const { body } = buildUpstreamRequest(
 { model: "gpt-5.5", messages: [{ role: "user", content }] },
 AZURE_MODEL,
 );
 expect(body.messages[0]?.content).toHaveLength(14_999);
 expect(body.messages[1]?.content).toBe("😀b");
 expect(body.messages.map((message) => message.content).join("")).toBe(content);
 expect(body.messages.some((message) => message.content.includes("�"))).toBe(false);
 });

 it("preserves image_url content parts for upstream vision models", () => {
    const image = {
      type: "image_url" as const,
      image_url: { url: "https://files.example.test/uploaded-photo.png" },
    };
    const { body } = buildUpstreamRequest(
      {
        model: "llama-4-scout",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Describe what this image means." },
              image,
            ],
          },
        ],
      },
      LMSYS_MODEL,
    );

    expect(body.messages).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "Describe what this image means." },
          image,
        ],
      },
    ]);
  });

  it("keeps edit tools available after a tool result on continue turns", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-5.5",
        messages: [
          { role: "user", content: "Fix the file" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: {
                  name: "read_file",
                  arguments: '{"filepath":"a.js"}',
                },
              },
            ],
          },
          { role: "tool", content: "file contents" },
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "read_file",
              parameters: { type: "object", properties: {} },
            },
          },
          {
            type: "function",
            function: {
              name: "edit_existing_file",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
      },
      AZURE_MODEL,
    );
    const systemNotes = body.messages
      .filter((message: { role: string }) => message.role === "system")
      .map((message) => String(message.content))
      .join("\n");
    expect(systemNotes).toMatch(
      /Continue the user's task|edit\/write\/replace/i,
    );
    expect(systemNotes).not.toMatch(/Do not announce another search/i);
  });

  it("allows another tool call after a tool result", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-5.5",
        messages: [
          { role: "user", content: "Create a report from search results." },
          { role: "tool", content: "Search results" },
        ],
        tools: [{ type: "function", function: { name: "create_document" } }],
      },
      AZURE_MODEL,
    );
    const transcript = body.messages
      .map((message) => String(message.content))
      .join("\n");
    expect(transcript).toContain(
      "If an edit/write/replace tool is still needed",
    );
    expect(transcript).not.toContain("Do not announce another search");
  });

  it("serializes tool-call history using the injected text protocol", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-5.5",
        messages: [
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: { name: "cron", arguments: '{"action":"add"}' },
              },
            ],
          },
          { role: "tool", content: "created" },
        ],
      },
      AZURE_MODEL,
    );
    expect(body.messages[0]?.content).toContain("<TOOL_CALL>");
    expect(body.messages[0]?.content).toContain("tool: cron");
    expect(body.messages[0]?.content).toContain('{"action":"add"}');
    expect(body.messages[1]?.content).toContain("<TOOL_RESULT>");
    expect(body.messages[1]?.content).toContain("created");
  });
  it("forwards the requested output token limit", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-5.5",
        messages: [{ role: "user", content: "write" }],
        max_tokens: 321,
      },
      AZURE_MODEL,
    );
    expect(body).toMatchObject({ maxTokens: 321 });
  });
  it("forwards GPT reasoning controls to the upstream body", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-5.5",
        messages: [{ role: "user", content: "solve this" }],
        reasoning_effort: "high",
        verbosity: "low",
      },
      AZURE_MODEL,
    );
    expect(body).toMatchObject({ reasoning_effort: "high", verbosity: "low" });
  });

  it("maps metadata.save -> !noSave and user -> chatId", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-5.5",
        messages: [{ role: "user", content: "hi" }],
        metadata: { save: true },
        user: "ck_existing_chat_id",
      },
      AZURE_MODEL,
    );
    expect(body.noSave).toBe(false);
    expect(body.chatId).toBe("ck_existing_chat_id");
  });
});

describe("buildUpstreamRequest — per-endpoint body shape", () => {
  const req: ChatCompletionRequest = {
    model: "x",
    messages: [{ role: "user", content: "hi" }],
  };

  it("azure: model is the provider/model slug, no apiKey", () => {
    const { endpoint, body } = buildUpstreamRequest(req, AZURE_MODEL);
    expect(endpoint).toBe("azure");
    expect(body).toMatchObject({ model: "openai/gpt-5.5" });
    expect(body).not.toHaveProperty("apiKey");
    expect(body).not.toHaveProperty("modelName");
  });

  it("perplexity: bare modelName + apiKey:null, no `model` field", () => {
    const { endpoint, body } = buildUpstreamRequest(req, PERPLEXITY_MODEL);
    expect(endpoint).toBe("perplexity");
    expect(body).not.toHaveProperty("model");
    expect(Object.keys(body).sort()).toEqual([
      "apiKey",
      "botId",
      "chatId",
      "fileUrl",
      "isRegenerate",
      "messages",
      "modelName",
      "noSave",
      "promptTemplate",
      "submissionId",
    ]);
    expect(body.submissionId).toMatch(/^[0-9a-f-]{36}$/i);
    expect(body).toMatchObject({
      modelName: "sonar-pro",
      apiKey: null,
      botId: "perplexity-sonar-pro",
    });
  });

  it("lmsys: modelName passed through in `model` + apiKey:null", () => {
    const { endpoint, body } = buildUpstreamRequest(req, LMSYS_MODEL);
    expect(endpoint).toBe("lmsys");
    expect(body).toMatchObject({
      model: "meta-llama/llama-4-scout-17b-16e-instruct",
      apiKey: null,
    });
    expect(body).not.toHaveProperty("modelName");
  });
});

describe("toEndpoint", () => {
  it("passes through the three known endpoints", () => {
    expect(toEndpoint("azure")).toBe("azure");
    expect(toEndpoint("perplexity")).toBe("perplexity");
    expect(toEndpoint("lmsys")).toBe("lmsys");
  });

  it("defaults unknown endpoint values to lmsys", () => {
    expect(toEndpoint("something-new")).toBe("lmsys");
    expect(toEndpoint("")).toBe("lmsys");
  });
});

describe("requiresNewPromptOnly", () => {
  it("treats Grok models as prompt-only agents", () => {
    const grok = { ...AZURE_MODEL, id: "xai/grok-4.7", modelName: "grok-4.7" };
    expect(requiresNewPromptOnly(grok)).toBe(true);
  });
});

describe("buildUpstreamHeaders", () => {
  it("matches the browser request headers without altering authorization", () => {
    const headers = new Headers(
      buildUpstreamHeaders("session-token", {
        UPSTREAM_ORIGIN: "https://web.example.test",
        UPSTREAM_REFERER: "https://web.example.test/",
      }),
    );
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("authorization")).toBe("Bearer session-token");
    expect(headers.get("origin")).toBe("https://web.example.test");
    expect(headers.get("referer")).toBe("https://web.example.test/");
    expect(headers.get("user-agent")).toContain("Chrome/155.0.0.0");
    expect(headers.get("sec-ch-ua")).toContain('"Brave"');
    expect(headers.get("accept-language")).toBe("en-US,en;q=0.9");
  });
});

describe("endpointUrl", () => {
  it("resolves each endpoint as a sibling of the configured azure URL", () => {
    expect(endpointUrl("azure", AZURE_BASE_URL)).toBe(AZURE_BASE_URL);
    expect(endpointUrl("perplexity", AZURE_BASE_URL)).toBe(
      "https://app.chatplayground.ai/api/chat/perplexity",
    );
    expect(endpointUrl("lmsys", AZURE_BASE_URL)).toBe(
      "https://app.chatplayground.ai/api/chat/lmsys",
    );
  });
});

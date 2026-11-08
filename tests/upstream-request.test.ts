import { describe, expect, it } from "vitest";
import { toEndpoint } from "../src/constants/endpoints";
import type { ModelEntry } from "../src/constants/models";
import type { ChatCompletionRequest } from "../src/types/openai";
import {
  buildUpstreamHeaders,
  buildUpstreamRequest,
  endpointUrl,
  formatHistoryText,
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

describe("formatHistoryText", () => {
  it("tags user turns and model responses, retaining tool results", () => {
    expect(
      formatHistoryText([
        { role: "user", content: "Latest TikTok food hacks" },
        { role: "assistant", content: "Here are some ideas." },
        { role: "tool", name: "web_search", tool_call_id: "call_1", content: "Found three results." },
        { role: "system", content: "Do not include system messages." },
      ]),
    ).toBe(
      `<USER>\nLatest TikTok food hacks\n</USER>\n\n` +
        `<RESPONSE>\nHere are some ideas.\n</RESPONSE>\n\n` +
        `<TOOL_RESULT>\nFound three results.\n</TOOL_RESULT>`,
    );
  });
});

describe("buildUpstreamRequest — field mapping", () => {
  it("preserves assistant tool instructions after agent search results", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-agent",
        tools: [
          {
            type: "function",
            function: {
              name: "web_search",
              parameters: {
                type: "object",
                properties: { query: { type: "string" } },
              },
            },
          },
        ],
        tool_choice: "auto",
        messages: [
          { role: "user", content: "Find the latest release." },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_search",
                type: "function",
                function: {
                  name: "web_search",
                  arguments: '{"query":"latest release"}',
                },
              },
            ],
          },
          {
            role: "tool",
            name: "web_search",
            content: "Release 3.2",
          },
        ],
      },
      AGENT_MODEL,
    );

    expect(body.messages[0]?.content).toContain("Find the latest release.");
    expect(body.messages[0]?.content).toContain("<TOOL_RESULT>");
    expect(body.messages[0]?.content).toContain("<TOOL_CALL>");
    expect(body.messages[0]?.content).toContain("web_search");
  });

  it("reasserts caller system instructions after tool results", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-agent",
        messages: [
          { role: "system", content: "Answer in exactly two bullet points." },
          { role: "user", content: "Summarize the search." },
          { role: "tool", name: "web_search", content: "Result A; Result B" },
        ],
        tools: [{ type: "function", function: { name: "web_search" } }],
      },
      AGENT_MODEL,
    );
    const transcript = String(body.messages[0]?.content);
    expect(transcript).toContain("Caller instructions (highest priority");
    expect(transcript).toContain("Answer in exactly two bullet points.");
    expect(transcript).toContain("Use every <TOOL_RESULT>");
  });

  it("folds full history for agent models and ignores the OpenAI user id", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-agent",
        user: "client-chat-id",
        messages: [
          { role: "system", content: "Follow these rules." },
          { role: "user", content: "Previous question" },
          { role: "assistant", content: "Previous answer" },
          { role: "user", content: "Current question" },
        ],
      },
      AGENT_MODEL,
    );
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]?.content).toContain("System: Follow these rules.");
    expect(body.messages[0]?.content).toContain("User: Previous question");
    expect(body.messages[0]?.content).toContain("Draft so far (do not repeat it):");
    expect(body.messages[0]?.content).toContain("Previous answer");
    expect(body.messages[0]?.content).toContain("User: Current question");
    expect(body.chatId).toBe("");
  });

  it("keeps current-turn tool results when continuing an agent prompt", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-agent",
        messages: [
          {
            role: "user",
            content: "Find the latest release and summarize it.",
          },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_search",
                type: "function",
                function: {
                  name: "web_search",
                  arguments: '{"query":"latest release"}',
                },
              },
            ],
          },
          {
            role: "tool",
            name: "web_search",
            content:
              '{"query":"latest release","results":[{"title":"Release 3.2","url":"https://example.test"}]}',
          },
        ],
      },
      AGENT_MODEL,
    );

    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]?.content).toContain(
      "Find the latest release and summarize it.",
    );
    expect(body.messages[0]?.content).toContain("<TOOL_RESULT>");
    expect(body.messages[0]?.content).toContain("Release 3.2");
    expect(body.messages[0]?.content).toContain("<TOOL_CALL>");
  });

  it("folds conversation history for agent models and ignores OpenAI user ids", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-agent",
        user: "client-chat-id",
        messages: [
          { role: "system", content: "System instructions." },
          { role: "user", content: "Previous question" },
          { role: "assistant", content: "Previous answer" },
          { role: "user", content: "Current prompt" },
        ],
      },
      AGENT_MODEL,
    );

    const transcript = String(body.messages[0]?.content);
    expect(transcript).toContain("System: System instructions.");
    expect(transcript).toContain("User: Previous question");
    expect(transcript).toContain("Draft so far (do not repeat it):");
    expect(transcript).toContain("Previous answer");
    expect(transcript).toContain("User: Current prompt");
    expect(body.chatId).toBe("");
    expect(body.noSave).toBe(true);
  });

  it("folds history and keeps Luna noSave enabled", () => {
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

    expect(body.messages).toHaveLength(1);
    const transcript = String(body.messages[0]?.content);
    expect(transcript).toContain("User: hello");
    expect(transcript).toContain("Draft so far (do not repeat it):\nHi!");
    expect(transcript).toContain("User: what is an API");
    expect(body.chatId).toBe("");
    expect(body.noSave).toBe(true);
    expect(body.submissionId).toMatch(/^[0-9a-f-]{36}$/i);
  });

  it("sends only the current user/tool turn when resuming a cached chat", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-5.5",
        messages: [
          { role: "system", content: "Keep these instructions." },
          { role: "user", content: "old prompt" },
          { role: "assistant", content: "old answer" },
          { role: "user", content: "new prompt" },
          { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "search", arguments: "{}" } }] },
          { role: "tool", name: "search", tool_call_id: "call_1", content: "result" },
        ],
      },
      AZURE_MODEL,
      { chatId: "c123456789012345678901", contextFilesOnly: true },
    );
    expect(body.chatId).toBe("c123456789012345678901");
    const transcript = String(body.messages[0]?.content);
    expect(transcript).toContain("Read tools.txt only for tool names and schemas");
    expect(transcript).toContain("Read history.txt only as conversation context");
    expect(transcript).toContain("new prompt");
    expect(transcript).toContain("<TOOL_RESULT>");
    expect(transcript).not.toContain("old answer");
    expect(transcript).not.toContain("old prompt");
  });

  it("folds history and does not resume an upstream chat from user id", () => {
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

    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]?.content).toContain("User: hello");
    expect(body.messages[0]?.content).toContain("User: what is an API");
    expect(body.chatId).toBe("");
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

  it("folds history to one message within the upstream size limit", () => {
    const content = "a".repeat(15_001);
    const { body } = buildUpstreamRequest(
      { model: "gpt-5.5", messages: [{ role: "user", content }] },
      AZURE_MODEL,
    );
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]?.content).toHaveLength(12_000);
    expect(body.messages[0]?.content).toContain("earlier context omitted");
    expect(body.messages[0]?.content).toContain(content.slice(-100));
  });

  it("puts the user request before injected tool instructions", () => {
    const { body } = buildUpstreamRequest(
      {
        model: "gpt-5.5",
        messages: [
          {
            role: "system",
            content: `[relay-tool-prompt-v1]\n=== TOOL POLICY ===\n${"tool rules ".repeat(700)}\n</AVAILABLE_TOOLS>`,
          },
          { role: "user", content: "Latest TikTok food hacks" },
        ],
      },
      AZURE_MODEL,
    );
    const content = String(body.messages[0]?.content);
    expect(content.indexOf("Latest TikTok food hacks")).toBeLessThan(
      content.indexOf("System:"),
    );
    expect(content).toContain("Latest TikTok food hacks");
  });

  it("caps oversized tool-policy prompts at the upstream message limit", () => {
    const content = `System rules\n=== TOOL POLICY ===\n${"tool schema ".repeat(2_000)}\n</AVAILABLE_TOOLS>\nUser: inspect the project`;
    const { body } = buildUpstreamRequest(
      { model: "gpt-5.5", messages: [{ role: "user", content }] },
      AZURE_MODEL,
    );
    const message = body.messages[0]?.content;
    expect(typeof message).toBe("string");
    expect((message as string).length).toBeLessThanOrEqual(12_000);
    expect(message).toContain("User: inspect the project");
    expect(message).toContain("earlier context omitted");
  });

  it("folds developer history into a user message and omits null assistant content", () => {
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
    expect(body.messages).toEqual([
      {
        role: "user",
        content: "System: Follow these rules.\n\nUser: hi",
      },
    ]);
  });

  it("keeps surrogate pairs intact while bounding a folded transcript", () => {
    const content = "a".repeat(14_999) + "😀" + "b";
    const { body } = buildUpstreamRequest(
      { model: "gpt-5.5", messages: [{ role: "user", content }] },
      AZURE_MODEL,
    );
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]?.content).toContain("😀b");
    expect(
      body.messages.some(
        (message) =>
          typeof message.content === "string" && message.content.includes("�"),
      ),
    ).toBe(false);
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
      .map((message) => String(message.content))
      .join("\n");
    expect(systemNotes).toContain("If another tool is still needed");
    expect(systemNotes).toContain("<TOOL_CALL> XML");
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
    expect(transcript).toContain("<TOOL_RESULT>");
    expect(transcript).toContain("Use this result to advance the original task.");
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
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]?.content).toContain("Assistant: <TOOL_CALL>");
    expect(body.messages[0]?.content).toContain("tool: cron");
    expect(body.messages[0]?.content).toContain('{"action":"add"}');
    expect(body.messages[0]?.content).toContain("Tool: <TOOL_RESULT>");
    expect(body.messages[0]?.content).toContain("created");
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

  it("maps metadata.save -> !noSave and leaves chatId stateless", () => {
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
    expect(body.chatId).toBe("");
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
    expect(Object.keys(body).toSorted()).toEqual([
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

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Hono } from "hono";
import { auth } from "../src/middleware/auth";
import { errorHandler } from "../src/middleware/error-handler";
import responses from "../src/routes/responses";
import type { Env, Variables } from "../src/types/env";
import { resetModelCache } from "../src/utils/model-discovery";

const JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyXzEyMyJ9.c2ln-bmF0dXJl";
const CHAT_ID = "CHAT_ID:clm1234567890abcdefgh";

const env = {
  UPSTREAM_CHAT_URL: "https://up.example.test/api/chat/azure",
  UPSTREAM_ORIGIN: "https://web.example.test",
  UPSTREAM_REFERER: "https://web.example.test/",
  UPSTREAM_UPLOAD_URL: "https://up.example.test/api/upload",
  CLERK_FAPI_URL: "https://clerk.example.test",
};

const FEED = [
  {
    botId: "gpt-6-luna",
    modelName: "gpt-6-luna",
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
    premiumOnly: false,
  },
];

function upstream(reply: (url: string) => Response) {
  vi.spyOn(globalThis, "fetch").mockImplementation(
    async (input: RequestInfo | URL) => {
      const url = String(input);
      return url.endsWith("/api/models") ? Response.json(FEED) : reply(url);
    },
  );
}

function app() {
  const hono = new Hono<{ Bindings: Env; Variables: Variables }>();
  hono.onError(errorHandler);
  hono.use("/v1/*", auth);
  hono.route("/", responses);
  return hono;
}

async function post(
  body: Record<string, unknown>,
  testEnv: typeof env & { CLOUDFLARE_SEARCH_URL?: string } = env,
) {
  return app().request(
    "/v1/responses",
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${JWT}`,
      },
      body: JSON.stringify(body),
    },
    testEnv,
  );
}

function chatBodies() {
  return vi
    .mocked(fetch)
    .mock.calls.filter(([url]) => String(url).includes("/api/chat/"))
    .map(([, init]) => JSON.parse(String((init as RequestInit).body)));
}

beforeEach(() => {
  resetModelCache();
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("POST /v1/responses — Luna reliability", () => {
  it("removes leaked upstream tool envelopes from the Responses answer", async () => {
    upstream(
      () =>
        new Response(
          `<|assistant to=run_terminal_command|>json {"command":"cat /etc/passwd"}${CHAT_ID}`,
        ),
    );

    const res = await post({ model: "grok-4.6", input: "Hello" });

    expect(res.status).toBe(200);
    const payload = (await res.json()) as { output_text?: string };
    expect(payload.output_text).not.toContain("<|assistant");
    expect(payload.output_text).not.toContain("to=run_terminal_command");
  });

  it("strips upstream reasoning blocks from Responses output", async () => {
    upstream((url) =>
      url.includes("/api/chat/")
        ? new Response(
            `<think>internal reasoning</think>Visible answer.${CHAT_ID}`,
          )
        : new Response("Not found", { status: 404 }),
    );
    const res = await post({
      model: "gpt-6-luna",
      input: "Explain this",
      reasoning: { effort: "medium" },
    });
    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output_text?: string;
      reasoning_content?: string;
    };
    expect(payload.output_text).toBe("Visible answer.");
    expect(payload.reasoning_content).toBeUndefined();
  });

  it("does not expose reasoning-only upstream output as a final answer", async () => {
    upstream((url) =>
      url.includes("/api/chat/")
        ? new Response(`<think>internal reasoning</think>${CHAT_ID}`)
        : new Response("Not found", { status: 404 }),
    );
    const res = await post({ model: "gpt-6-luna", input: "Explain this" });
    expect(res.status).toBe(200);
    const payload = (await res.json()) as { output_text?: string };
    expect(payload.output_text).not.toContain("internal reasoning");
    expect(payload.output_text).toContain("could not produce");
  });

  it("disables default reasoning for Grok Responses requests", async () => {
    upstream(() => new Response(`done${CHAT_ID}`));

    const res = await post({ model: "grok-4.6", input: "Hello" });

    expect(res.status).toBe(200);
    expect(chatBodies()[0]?.reasoning_effort).toBe("none");
    expect(chatBodies()[0]?.maxTokens).toBe(1024);
  });

  it("forces reasoning_effort none when Luna has tools", async () => {
    upstream(() => new Response(`done${CHAT_ID}`));
    const res = await post({
      model: "gpt-6-luna",
      input: "Search for the latest AI news",
      tools: [
        {
          type: "function",
          name: "web_search",
          parameters: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
          },
        },
      ],
    });
    expect(res.status).toBe(200);
    const bodies = chatBodies();
    expect(bodies.length).toBeGreaterThan(0);
    expect(bodies[0].reasoning_effort).toBe("none");
  });

  it.each([false, true])(
    "leaves document tool selection to the model (stream=%s)",
    async (stream) => {
      upstream((url) =>
        url.includes("/api/chat/")
          ? new Response(
              `<TOOL_CALL>\ntool: create_document\nparams: {"title":"Report"}\n</TOOL_CALL>${CHAT_ID}`,
            )
          : new Response("Not found", { status: 404 }),
      );
      const res = await post({
        model: "gpt-6-luna",
        input: "Create a document about our project.",
        stream,
        tool_choice: "auto",
        tools: [
          {
            type: "function",
            name: "create_document",
            parameters: {
              type: "object",
              properties: { title: { type: "string" } },
              required: ["title"],
            },
          },
        ],
      });
      expect(res.status).toBe(200);
      const prompt = JSON.stringify(chatBodies()[0]?.messages);
      expect(prompt).toContain(
        "Call a tool only when its result is required. Otherwise answer directly.",
      );
      expect(prompt).not.toContain("You must call exactly one listed tool.");
      if (stream) {
        const events = await res.text();
        expect(events).toContain("response.function_call_arguments");
        expect(events).toContain("create_document");
      } else {
        const payload = (await res.json()) as {
          output?: Array<{ type?: string; name?: string }>;
        };
        expect(payload.output?.[0]).toMatchObject({
          type: "function_call",
          name: "create_document",
        });
      }
    },
  );

  it.each(["web_search", "web_search_preview"])(
    "falls back after 500 ms to a caller-advertised search function for %s",
    async (hostedType) => {
      let aborts = 0;
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
        const url = String(input);
        if (url.endsWith("/api/models")) return Response.json(FEED);
        if (url.includes("/api/chat/")) {
          return new Response(`Grounded answer.${CHAT_ID}`);
        }
        if (url.includes("api.you.com")) {
          return Response.json({
            results: [
              {
                title: "News",
                url: "https://example.test/news",
                snippet: "AI news.",
              },
            ],
          });
        }
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            "abort",
            () => {
              aborts++;
              reject(new Error("aborted"));
            },
            { once: true },
          );
        });
      });
      const started = Date.now();
      const res = await post(
        {
          model: "gpt-6-luna",
          input: "Find the latest AI news",
          tools: [
            { type: hostedType },
            {
              type: "function",
              name: "web_browse",
              parameters: {
                type: "object",
                properties: { search_query: { type: "string" } },
                required: ["search_query"],
              },
            },
          ],
        },
        { ...env, CLOUDFLARE_SEARCH_URL: "https://search.example.test" },
      );
      expect(res.status).toBe(200);
      expect(Date.now() - started).toBeGreaterThanOrEqual(450);
      expect(Date.now() - started).toBeLessThan(2000);
      const payload = (await res.json()) as {
        output?: Array<{ type?: string }>;
        output_text?: string;
      };
      expect(
        payload.output?.some((item) => item.type === "web_search_call"),
      ).toBe(true);
      expect(
        payload.output?.some((item) => item.type === "function_call"),
      ).toBe(true);
      expect(payload.output_text).toBe("");
      expect(aborts).toBeGreaterThan(0);
    },
  );

  it("returns a search function on failure without an OpenAI search block or status text", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/models")) return Response.json(FEED);
      if (url.includes("/api/chat/")) {
        return new Response(`Grounded answer.${CHAT_ID}`);
      }
      return new Response("Unavailable", { status: 503 });
    });

    const res = await post(
      {
        model: "gpt-6-luna",
        input: "Find current AI news",
        stream: true,
        tools: [
          { type: "web_search" },
          {
            type: "function",
            name: "web_browse",
            parameters: {
              type: "object",
              properties: { query: { type: "string" } },
              required: ["query"],
            },
          },
        ],
      },
      { ...env, CLOUDFLARE_SEARCH_URL: "https://search.example.test" },
    );
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("web_browse");
    expect(body).not.toContain("web_search_call");
    expect(body).not.toContain("OpenAI web search source");
  });

  it("returns the client search function when a hosted call fails after model selection", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/models")) return Response.json(FEED);
      if (url.includes("/api/chat/")) {
        return new Response(
          `<TOOL_CALL>\\ntool: web_search\\nparams: {"query":"help me find something"}\\n</TOOL_CALL>${CHAT_ID}`,
        );
      }
      return new Response("Unavailable", { status: 503 });
    });

    const res = await post(
      {
        model: "gpt-6-luna",
        input: "help me find something",
        tools: [
          { type: "web_search" },
          {
            type: "function",
            name: "search_news",
            parameters: {
              type: "object",
              properties: { query: { type: "string" } },
              required: ["query"],
            },
          },
        ],
      },
      { ...env, CLOUDFLARE_SEARCH_URL: "https://search.example.test" },
    );

    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output?: Array<{ type?: string; name?: string }>;
    };
    expect(
      payload.output?.some((item) => item.type === "web_search_call"),
    ).toBe(false);
    expect(payload.output?.some((item) => item.type === "function_call")).toBe(
      true,
    );
  });

  it("returns a caller's custom search function without starting hosted search", async () => {
    const requests: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requests.push(url);
      if (url.endsWith("/api/models")) return Response.json(FEED);
      if (url.includes("/api/chat/")) {
        return new Response(
          `<TOOL_CALL>\ntool: web_browse\nparams: {"query":"latest AI news"}\n</TOOL_CALL>${CHAT_ID}`,
        );
      }
      return new Response("Unexpected", { status: 500 });
    });
    const res = await post({
      model: "gpt-6-luna",
      input: "Find latest AI news",
      tools: [
        {
          type: "function",
          name: "web_browse",
          parameters: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
          },
        },
      ],
    });
    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output?: Array<{ type?: string; name?: string }>;
    };
    expect(payload.output?.[0]).toMatchObject({
      type: "function_call",
      name: "web_browse",
    });
    expect(
      requests.every(
        (url) => url.endsWith("/api/models") || url.includes("/api/chat/"),
      ),
    ).toBe(true);
  });

  it("does not impose the fallback deadline without a caller-owned search tool", async () => {
    let chats = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/models")) return Response.json(FEED);
      if (url.includes("/api/chat/")) {
        chats++;
        return new Response(
          chats === 1 ? `Searching.${CHAT_ID}` : `Found the news.${CHAT_ID}`,
        );
      }
      if (url.startsWith("https://search.example.test/")) {
        await new Promise((resolve) => setTimeout(resolve, 750));
        return Response.json({
          results: [
            {
              title: "News",
              url: "https://example.test/news",
              snippet: "AI news.",
            },
          ],
        });
      }
      return new Response("Unavailable", { status: 503 });
    });
    const res = await post(
      {
        model: "gpt-6-luna",
        input: "Find the latest AI news",
        tools: [{ type: "web_search" }],
      },
      { ...env, CLOUDFLARE_SEARCH_URL: "https://search.example.test" },
    );
    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output?: Array<{ type?: string }>;
      output_text?: string;
    };
    expect(
      payload.output?.some((item) => item.type === "web_search_call"),
    ).toBe(true);
    expect(payload.output_text).toContain("Found the news");
    expect(chats).toBe(2);
  });

  it("does not delegate when multiple client search tools are ambiguous", async () => {
    let chats = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/models")) return Response.json(FEED);
      if (url.includes("/api/chat/")) {
        chats++;
        return new Response(
          chats === 1
            ? `<TOOL_CALL>\ntool: web_search\nparams: {"query":"AI news"}\n</TOOL_CALL>${CHAT_ID}`
            : `Found the news.${CHAT_ID}`,
        );
      }
      if (url.startsWith("https://search.example.test/")) {
        await new Promise((resolve) => setTimeout(resolve, 750));
        return Response.json({
          results: [
            {
              title: "News",
              url: "https://example.test/news",
              snippet: "AI news.",
            },
          ],
        });
      }
      return new Response("Unavailable", { status: 503 });
    });
    const searchFunction = (name: string) => ({
      type: "function",
      name,
      parameters: {
        type: "object",
        properties: { query: { type: "string" } },
        required: ["query"],
      },
    });
    const res = await post(
      {
        model: "gpt-6-luna",
        input: "Find the latest AI news",
        tools: [
          { type: "web_search" },
          searchFunction("web_browse"),
          searchFunction("search_news"),
        ],
      },
      { ...env, CLOUDFLARE_SEARCH_URL: "https://search.example.test" },
    );
    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output?: Array<{ type?: string }>;
      output_text?: string;
    };
    expect(
      payload.output?.some((item) => item.type === "web_search_call"),
    ).toBe(true);
    expect(payload.output?.some((item) => item.type === "function_call")).toBe(
      false,
    );
    expect(payload.output_text).toContain("Found the news");
  });

  it("does not expose OpenAI hosted search to Grok for a basic question", async () => {
    upstream(
      () =>
        new Response(
          `Binary search repeatedly halves a sorted range.${CHAT_ID}`,
        ),
    );

    const res = await post({
      model: "grok-4.6",
      input: "Explain binary search.",
      tools: [{ type: "web_search" }],
    });

    expect(res.status).toBe(200);
    const payload = (await res.json()) as { output_text?: string };
    expect(payload.output_text).toContain("Binary search");
    expect(chatBodies()).toHaveLength(1);
    expect(chatBodies()[0]?.tools).toBeUndefined();
    expect(JSON.stringify(chatBodies()[0]?.messages)).not.toContain(
      "web_search",
    );
  });

  it("executes an OpenAI hosted web-search request instead of returning a client function call", async () => {
    let chatCalls = 0;
    upstream((url) => {
      if (url.includes("/api/chat/")) {
        chatCalls += 1;
        return new Response(
          chatCalls === 1
            ? `<TOOL_CALL>\ntool: web_search\nparams: {"query":"latest AI news"}\n</TOOL_CALL>${CHAT_ID}`
            : `The latest AI news is available.${CHAT_ID}`,
        );
      }
      if (url.includes("api.you.com")) {
        return Response.json({
          results: [
            {
              title: "News",
              url: "https://example.test/news",
              snippet: "Latest AI news.",
            },
          ],
        });
      }
      return new Response("Not found", { status: 404 });
    });

    const res = await post({
      model: "gpt-6-luna",
      input: "latest AI news",
      tools: [{ type: "web_search_preview" }],
    });

    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output_text?: string;
      output?: Array<{ type?: string }>;
    };
    expect(payload.output_text).toContain("latest AI news");
    expect(
      payload.output?.some((item) => item.type === "web_search_call"),
    ).toBe(true);
    expect(payload.output?.some((item) => item.type === "function_call")).toBe(
      false,
    );
    expect(chatCalls).toBe(2);
  });

  it("returns configured Cloudflare Web Search API items in the OpenAI search block", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/models")) return Response.json(FEED);
      if (url.includes("/api/chat/")) {
        return new Response(`The results cover current AI news.${CHAT_ID}`);
      }
      if (url.startsWith("https://search.example.test/")) {
        return Response.json({
          items: [
            {
              title: "AI News",
              url: "https://example.test/ai-news",
              description: "A report about current AI news.",
            },
          ],
          metadata: { query: "latest AI news" },
        });
      }
      return new Response("Unexpected request", { status: 500 });
    });

    const res = await post(
      {
        model: "gpt-6-luna",
        input: "latest AI news",
        tools: [{ type: "web_search" }],
      },
      { ...env, CLOUDFLARE_SEARCH_URL: "https://search.example.test" },
    );

    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output?: Array<{
        type?: string;
        results?: Array<{ title?: string; url?: string }>;
      }>;
    };
    const searchCall = payload.output?.find(
      (item) => item.type === "web_search_call",
    );
    expect(searchCall?.results).toEqual([
      {
        title: "AI News",
        url: "https://example.test/ai-news",
        snippet: "A report about current AI news.",
      },
    ]);
  });

  it("performs hosted search for an explicit search prompt without current-events keywords", async () => {
    let chats = 0;
    upstream((url) => {
      if (url.includes("/api/chat/")) {
        chats++;
        return new Response(
          chats === 1
            ? `I will search.${CHAT_ID}`
            : `Found the guide.${CHAT_ID}`,
        );
      }
      if (url.includes("search"))
        return Response.json({
          results: [
            {
              title: "Guide",
              url: "https://example.test/guide",
              snippet: "A guide.",
            },
          ],
        });
      return new Response("Not found", { status: 404 });
    });
    const res = await post({
      model: "gpt-6-luna",
      input: "Search for a beginner guide to pottery",
      tools: [{ type: "web_search" }],
    });
    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output?: Array<{ type?: string }>;
      output_text?: string;
    };
    expect(
      payload.output?.some((item) => item.type === "web_search_call"),
    ).toBe(true);
    expect(payload.output_text).toContain("Found the guide");
    expect(chats).toBe(2);
  });

  it("preserves ordinary function calls for the Responses client", async () => {
    upstream((url) =>
      url.includes("/api/chat/")
        ? new Response(
            `<TOOL_CALL>\ntool: get_weather\nparams: {"city":"Tokyo"}\n</TOOL_CALL>${CHAT_ID}`,
          )
        : new Response("Not found", { status: 404 }),
    );

    const res = await post({
      model: "gpt-6-luna",
      input: "Get Tokyo weather",
      tools: [
        {
          type: "function",
          name: "get_weather",
          parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      ],
    });

    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output?: Array<{ type?: string; name?: string }>;
    };
    expect(payload.output?.[0]).toMatchObject({
      type: "function_call",
      name: "get_weather",
    });
  });

  it("does not invoke tools when tool_choice is none", async () => {
    upstream((url) =>
      url.includes("/api/chat/")
        ? new Response(`Direct answer.${CHAT_ID}`)
        : new Response("Not found", { status: 404 }),
    );

    const res = await post({
      model: "gpt-6-luna",
      input: "latest AI news",
      tool_choice: "none",
      tools: [{ type: "web_search_preview" }],
    });

    expect(res.status).toBe(200);
    expect(chatBodies()[0]?.messages[0]?.content).not.toContain("<TOOL_CALL>");
    const payload = (await res.json()) as { output?: Array<{ type?: string }> };
    expect(
      payload.output?.some(
        (item) =>
          item.type === "web_search_call" || item.type === "function_call",
      ),
    ).toBe(false);
  });

  it("does not return invalid TOOL_CALL XML after a failed gate retry", async () => {
    let chats = 0;
    upstream(() => {
      chats += 1;
      return new Response(
        `<TOOL_CALL> tool: single_find_and_replace params: {"filepath":"utils/searchBrave.js","old_string":"html.split("\n")","new_string":"html.split(\\n)"} </TOOL_CALL>${CHAT_ID}`,
      );
    });
    const res = await post({
      model: "gpt-6-luna",
      input: "Fix searchBrave.js",
      tools: [
        {
          type: "function",
          name: "single_find_and_replace",
          parameters: {
            type: "object",
            properties: {
              filepath: { type: "string" },
              old_string: { type: "string" },
              new_string: { type: "string" },
            },
            required: ["filepath", "old_string", "new_string"],
          },
        },
      ],
    });
    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output_text?: string;
      choices?: Array<{ message?: { content?: string } }>;
      output?: Array<{ type?: string; arguments?: string }>;
    };
    const text =
      payload.output_text ?? payload.choices?.[0]?.message?.content ?? "";
    expect(text).not.toMatch(/<TOOL_CALL\b/i);
    expect(text).toMatch(/invalid|retry/i);
    expect(
      payload.output?.some((item) => item.type === "function_call"),
    ).toBeFalsy();
    expect(chats).toBeGreaterThan(1);
  });

  it("makes Luna answer directly from hosted search results after a verification refusal", async () => {
    let chats = 0;
    upstream((url) => {
      if (url.includes("/api/chat/")) {
        chats += 1;
        return new Response(
          (chats === 1
            ? "I can't verify that, but the search results say the event occurred."
            : "Yes. The event occurred, according to the search result.") +
            CHAT_ID,
        );
      }
      if (url.includes("api.you.com")) {
        return Response.json({
          results: [
            {
              title: "Event",
              url: "https://example.test/event",
              snippet: "The event occurred.",
            },
          ],
        });
      }
      return new Response("Not found", { status: 404 });
    });

    const res = await post({
      model: "gpt-6-luna",
      input: "Verify whether the event occurred.",
      tools: [{ type: "web_search" }],
    });

    expect(res.status).toBe(200);
    const payload = (await res.json()) as { output_text?: string };
    expect(payload.output_text).toContain(
      "Yes. The event occurred, according to the search result.",
    );
    expect(chats).toBe(2);
    const repair = JSON.stringify(chatBodies()[1]?.messages);
    expect(repair).toContain(
      "Treat this result as evidence for the user's original question",
    );
    expect(repair).toContain(
      "do not say you cannot verify it or merely list the result",
    );
  });

  it("renders hosted search as an Agora OpenAI Search output item", async () => {
    let chatCalls = 0;
    upstream((url) => {
      if (url.includes("/api/chat/")) {
        chatCalls++;
        return new Response(
          chatCalls === 1
            ? `<TOOL_CALL>\ntool: web_search\nparams: {"query":"latest AI news"}\n</TOOL_CALL>${CHAT_ID}`
            : `The latest AI news is available.${CHAT_ID}`,
        );
      }
      if (url.includes("api.you.com")) {
        return Response.json({
          results: [
            {
              title: "News",
              url: "https://example.test/news",
              snippet: "Latest AI news.",
            },
          ],
        });
      }
      return new Response("Not found", { status: 404 });
    });

    const res = await post({
      model: "gpt-6-luna",
      input: "latest AI news",
      tools: [{ type: "web_search" }],
    });

    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output?: Array<{
        type?: string;
        status?: string;
        action?: { type?: string; query?: string };
        results?: Array<{ url?: string }>;
      }>;
      output_text?: string;
    };
    const hosted = payload.output?.find(
      (item) => item.type === "web_search_call",
    );
    expect(hosted).toMatchObject({
      status: "completed",
      action: { type: "search", query: "latest AI news" },
      results: [{ url: "https://example.test/news" }],
    });
    expect(payload.output_text).not.toContain("OpenAI web search results");
  });

  it("uses the client-advertised search name and schema after hosted timeout", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/api/models")) return Response.json(FEED);
      if (url.includes("/api/chat/")) return new Response(`Answer.${CHAT_ID}`);
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener(
          "abort",
          () => reject(new Error("aborted")),
          { once: true },
        );
      });
    });

    const res = await post(
      {
        model: "grok-4.6",
        input: "Find the latest AI news",
        tools: [
          { type: "web_search_preview" },
          {
            type: "function",
            name: "duckduckgo_lookup",
            parameters: {
              type: "object",
              properties: { search_query: { type: "string" } },
              required: ["search_query"],
            },
          },
        ],
      },
      { ...env, CLOUDFLARE_SEARCH_URL: "https://search.example.test" },
    );

    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output?: Array<{
        type?: string;
        name?: string;
        arguments?: string;
      }>;
    };
    expect(
      payload.output?.some((item) => item.type === "web_search_call"),
    ).toBe(true);
    const fallback = payload.output?.find(
      (item) => item.type === "function_call",
    );
    expect(fallback?.name).toBe("duckduckgo_lookup");
    expect(JSON.parse(fallback?.arguments ?? "{}")).toEqual({
      search_query: "Find the latest AI news",
    });
  });

  it("does not execute hosted search when OpenAI Search is absent", async () => {
    let chatCalls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/models")) return Response.json(FEED);
      if (url.includes("/api/chat/")) {
        chatCalls++;
        return new Response(
          `<TOOL_CALL>\ntool: local_search\nparams: {"query":"AI news"}\n</TOOL_CALL>${CHAT_ID}`,
        );
      }
      return new Response("Unexpected hosted search", { status: 500 });
    });

    const res = await post({
      model: "gpt-6-luna",
      input: "Find the latest AI news",
      tools: [
        {
          type: "function",
          name: "local_search",
          parameters: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
          },
        },
      ],
    });

    expect(res.status).toBe(200);
    expect(chatCalls).toBe(1);
    const payload = (await res.json()) as {
      output?: Array<{ type?: string; name?: string }>;
    };
    expect(payload.output?.[0]).toMatchObject({
      type: "function_call",
      name: "local_search",
    });
    expect(
      payload.output?.some((item) => item.type === "web_search_call"),
    ).toBe(false);
  });

  it("retries once after a plan-only reply when a tool result is present", async () => {
    let chats = 0;
    upstream(() => {
      chats += 1;
      return new Response(
        (chats === 1
          ? "I’ll check current coverage and distinguish the popular hacks."
          : "The popular hack is freezing grapes.") + CHAT_ID,
      );
    });
    const res = await post({
      model: "gpt-6-luna",
      input: [
        { role: "user", content: "What food hacks are popular?" },
        {
          type: "function_call",
          call_id: "call_1",
          name: "web_search",
          arguments: '{"query":"food hacks"}',
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: "Freezing grapes is the popular hack.",
        },
      ],
      tools: [
        {
          type: "function",
          name: "web_search",
          parameters: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
          },
        },
      ],
    });
    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      output_text?: string;
      choices?: Array<{ message?: { content?: string } }>;
    };
    const text =
      payload.output_text ?? payload.choices?.[0]?.message?.content ?? "";
    expect(text).toContain("freezing grapes");
    expect(chats).toBeGreaterThan(1);
  });
});

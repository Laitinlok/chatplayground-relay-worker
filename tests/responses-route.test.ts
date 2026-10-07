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
];

function upstream(reply: (url: string) => Response) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
    const url = String(input);
    return url.endsWith("/api/models") ? Response.json(FEED) : reply(url);
  });
}

function app() {
  const hono = new Hono<{ Bindings: Env; Variables: Variables }>();
  hono.onError(errorHandler);
  hono.use("/v1/*", auth);
  hono.route("/", responses);
  return hono;
}

async function post(body: Record<string, unknown>) {
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
    env,
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
  it("forces reasoning_effort none when Luna has tools", async () => {
    upstream(() => new Response(`done${CHAT_ID}`));
    const res = await post({
      model: "gpt-6-luna",
      input: "Search for the latest AI news",
      tools: [{ type: "function", name: "web_search", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } }],
    });
    expect(res.status).toBe(200);
    const bodies = chatBodies();
    expect(bodies.length).toBeGreaterThan(0);
    expect(bodies[0].reasoning_effort).toBe("none");
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
    const text = payload.output_text ?? payload.choices?.[0]?.message?.content ?? "";
    expect(text).not.toMatch(/<TOOL_CALL\b/i);
    expect(text).toMatch(/invalid|retry/i);
    expect(payload.output?.some((item) => item.type === "function_call")).toBeFalsy();
    expect(chats).toBeGreaterThan(1);
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
          arguments: "{\"query\":\"food hacks\"}",
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: "Freezing grapes is the popular hack.",
        },
      ],
      tools: [{ type: "function", name: "web_search", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } }],
    });
    expect(res.status).toBe(200);
    const payload = await res.json() as { output_text?: string; choices?: Array<{ message?: { content?: string } }> };
    const text = payload.output_text ?? payload.choices?.[0]?.message?.content ?? "";
    expect(text).toContain("freezing grapes");
    expect(chats).toBeGreaterThan(1);
  });
});

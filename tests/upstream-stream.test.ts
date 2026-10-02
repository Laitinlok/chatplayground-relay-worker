import { describe, expect, it } from "vitest";
import {
  formatCitations,
  formatThinkTags,
  inlineCitationLinks,
  parseTrailers,
  splitReasoningContent,
  streamUpstreamAsOpenAI,
  streamUpstreamWithToolShim,
} from "../src/utils/upstream-stream";

const CUID = "cabcdefghijklmnopqrstuvwx"; // 24 chars after the leading "c"
const URLS = [
  "https://example.com/1",
  "https://example.com/2",
  "https://example.com/3",
];

describe("parseTrailers", () => {
  it("returns the buffer unchanged when no trailer is present", () => {
    const out = parseTrailers("hello world");
    expect(out).toEqual({
      content: "hello world",
      chatId: null,
      citations: [],
    });
  });

  it("strips the CHAT_ID sentinel and extracts the cuid", () => {
    const out = parseTrailers(`prose\nCHAT_ID:${CUID}`);
    expect(out.content).toBe("prose\n");
    expect(out.chatId).toBe(CUID);
    expect(out.citations).toEqual([]);
  });

  it("strips the perplexity CITATIONS payload and parses URLs", () => {
    const json = JSON.stringify(URLS);
    const out = parseTrailers(`answer text CITATIONS:${json}`);
    expect(out.content).toBe("answer text ");
    expect(out.citations).toEqual(URLS);
    expect(out.chatId).toBeNull();
  });

  it("handles CITATIONS followed by CHAT_ID (order-tolerant)", () => {
    const json = JSON.stringify(URLS);
    const out = parseTrailers(`answer CITATIONS:${json}CHAT_ID:${CUID}`);
    expect(out.content).toBe("answer ");
    expect(out.citations).toEqual(URLS);
    expect(out.chatId).toBe(CUID);
  });

  it("handles CHAT_ID followed by CITATIONS (order-tolerant)", () => {
    const json = JSON.stringify(URLS);
    const out = parseTrailers(`answer CHAT_ID:${CUID}CITATIONS:${json}`);
    expect(out.content).toBe("answer ");
    expect(out.citations).toEqual(URLS);
    expect(out.chatId).toBe(CUID);
  });

  it("leaves a malformed CITATIONS payload untouched", () => {
    const malformed = "answer CITATIONS:[this is not, valid json]";
    const out = parseTrailers(malformed);
    expect(out.citations).toEqual([]);
    // Malformed payload is left in content rather than silently corrupted.
    expect(out.content).toBe(malformed);
  });
});

describe("splitReasoningContent", () => {
  it("separates think text from the final answer", () => {
    expect(splitReasoningContent("<think>step one</think>answer")).toEqual({
      content: "answer",
      reasoningContent: "step one",
    });
  });

  it("decodes escaped tags and accepts reasoning aliases", () => {
    expect(
      splitReasoningContent(
        "&lt;analysis&gt;first&lt;/analysis&gt;<reasoning>second</reasoning>answer",
      ),
    ).toEqual({ content: "answer", reasoningContent: "firstsecond" });
  });

  it("preserves ordinary text when no reasoning tags are present", () => {
    expect(splitReasoningContent("plain answer")).toEqual({
      content: "plain answer",
      reasoningContent: "",
    });
  });

  it("recovers DeepSeek reasoning when only the closing tag is emitted", () => {
    expect(splitReasoningContent("private reasoning</think>answer")).toEqual({
      content: "answer",
      reasoningContent: "private reasoning",
    });
  });
});

async function readStream(stream: ReadableStream<Uint8Array>): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let result = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (value) result += decoder.decode(value, { stream: true });
  }
  return result + decoder.decode();
}

function upstreamBody(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

function ssePayloads(streamText: string): Array<Record<string, unknown>> {
  return streamText
    .split("\n\n")
    .filter((event) => event.startsWith("data: {"))
    .map((event) => JSON.parse(event.slice(6)) as Record<string, unknown>);
}

describe("Agora reasoning SSE compatibility", () => {
  const meta = { id: "chatcmpl-test", model: "test-model", created: 1 };

  it("emits reasoning_content separately when tags cross upstream chunks", async () => {
    const output = await readStream(
      streamUpstreamAsOpenAI(
        upstreamBody(["<thi", "nk>private reasoning</th", "ink>public answer"]),
        meta,
      ),
    );
    const deltas = ssePayloads(output).map(
      (payload) =>
        (payload.choices as Array<{ delta: Record<string, unknown> }>)[0]
          ?.delta ?? {},
    );
    expect(
      deltas.some((delta) => delta.reasoning_content === "private reasoning"),
    ).toBe(true);
    expect(deltas.some((delta) => delta.content === "public answer")).toBe(
      true,
    );
    expect(output).not.toContain("<think>");
  });

  it("preserves reasoning_content on the tool-aware streaming path", async () => {
    const output = await readStream(
      streamUpstreamWithToolShim(
        upstreamBody(["<think>private</think>answer"]),
        meta,
      ),
    );
    const deltas = ssePayloads(output).map(
      (payload) =>
        (payload.choices as Array<{ delta: Record<string, unknown> }>)[0]
          ?.delta ?? {},
    );
    expect(deltas.some((delta) => delta.reasoning_content === "private")).toBe(
      true,
    );
    expect(deltas.some((delta) => delta.content === "answer")).toBe(true);
  });
});

describe("formatThinkTags", () => {
  it("retains a normalized inline fallback for legacy clients", () => {
    expect(formatThinkTags("<think>step one</think>answer")).toBe(
      "\n<think>\nstep one\n</think>\nanswer",
    );
  });
});
describe("formatCitations", () => {
  it("returns an empty string for no citations", () => {
    expect(formatCitations([])).toBe("");
  });

  it("produces a Markdown sources block with clickable links", () => {
    const out = formatCitations(URLS);
    expect(out).toBe(
      "\n\n---\n**Sources**\n\n" +
        "1. [example.com](https://example.com/1)\n" +
        "2. [example.com](https://example.com/2)\n" +
        "3. [example.com](https://example.com/3)",
    );
  });
});

describe("inlineCitationLinks", () => {
  it("returns text unchanged when there are no citations", () => {
    expect(inlineCitationLinks("answer [1] more", [])).toBe("answer [1] more");
  });

  it("rewrites known [N] markers as Markdown links with escaped brackets", () => {
    const out = inlineCitationLinks("see [1] and [2].", URLS);
    expect(out).toBe(
      "see [\\[1\\]](https://example.com/1) and [\\[2\\]](https://example.com/2).",
    );
  });

  it("rewrites adjacent markers like [7][2] independently", () => {
    const out = inlineCitationLinks("evidence [3][1]", URLS);
    expect(out).toBe(
      "evidence [\\[3\\]](https://example.com/3)[\\[1\\]](https://example.com/1)",
    );
  });

  it("leaves out-of-range markers as literal text", () => {
    const out = inlineCitationLinks("a [9] z", URLS);
    expect(out).toBe("a [9] z");
  });
});

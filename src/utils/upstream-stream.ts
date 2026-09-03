import type {
  ChatCompletionChunk,
  ChatCompletionChunkDelta,
} from "../types/openai";
import { tryParseRelayToolCall, type OpenAITool, type ShimToolCall } from "./tool-shim";

// chatplayground appends `CHAT_ID:<cuid>` at the very end of the stream as a
// sentinel. CUID format: `c` + ≥20 chars of [a-z0-9]. We strip it before
// emitting content.
const SENTINEL_RE = /CHAT_ID:(c[a-z0-9]{20,})$/;

// perplexity emits its citation list as a trailing chunk of the form
// `CITATIONS:["url1","url2",...]`. The web client parses it as structured
// citations; we strip it from the prose and re-format as Markdown so
// OpenAI-compatible clients render proper links.
const CITATIONS_RE = /CITATIONS:(\[[\s\S]*?\])/;
const REASONING_TAG_PATTERN = "<(\\/?)\\s*(think|thinking|reasoning|analysis|thought)\\s*>";
const ESCAPED_REASONING_TAG_PATTERN =
  "&lt;(\\/?)\\s*(think|thinking|reasoning|analysis|thought)\\s*&gt;";

export interface ParsedUpstream {
  content: string;
  chatId: string | null;
  citations: readonly string[];
}

/** Read entire upstream body, strip trailers, return content + chatId + citations. */
export async function collectUpstream(
  body: ReadableStream<Uint8Array>,
): Promise<ParsedUpstream> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (value) buf += decoder.decode(value, { stream: true });
  }
  buf += decoder.decode();
  return parseTrailers(buf);
}

/**
 * Strip trailing CHAT_ID sentinel and perplexity CITATIONS payload from a
 * fully-buffered upstream response. Order-tolerant: CITATIONS is matched
 * with no anchor first, so either ordering (CITATIONS-then-CHAT_ID or
 * CHAT_ID-then-CITATIONS) is handled.
 */
export function parseTrailers(buf: string): ParsedUpstream {
  let working = buf;
  let citations: readonly string[] = [];

  const cm = CITATIONS_RE.exec(working);
  if (cm?.[1]) {
    const parsed = safeParseStringArray(cm[1]);
    if (parsed) {
      citations = parsed;
      working =
        working.slice(0, cm.index) + working.slice(cm.index + cm[0].length);
    }
  }

  const sm = SENTINEL_RE.exec(working);
  let chatId: string | null = null;
  if (sm) {
    chatId = sm[1] ?? null;
    working = working.slice(0, sm.index);
  }

  return { content: working, chatId, citations };
}

function safeParseStringArray(json: string): readonly string[] | null {
  try {
    const value: unknown = JSON.parse(json);
    if (!Array.isArray(value)) return null;
    return value.filter((u): u is string => typeof u === "string");
  } catch {
    return null;
  }
}

/** Markdown sources block appended at the end of an assistant message. */
export function formatCitations(citations: readonly string[]): string {
  if (citations.length === 0) return "";
  // Wrap each URL as a Markdown link so renderers that don't autolink bare
  // URLs still produce a clickable Sources list. Visible text stays the URL.
  const lines = citations.map((url, i) => `${i + 1}. [${url}](${url})`);
  return `\n\n---\n**Sources**\n\n${lines.join("\n")}`;
}

const INLINE_CITATION_RE = /\[(\d+)\]/g;

/**
 * Rewrite `[N]` citation markers in `text` to Markdown links pointing at the
 * Nth citation URL. Used only on the non-streaming path — the streaming path
 * flushes prose live and can only append a sources block at the end.
 */
export function inlineCitationLinks(
  text: string,
  citations: readonly string[],
): string {
  if (citations.length === 0) return text;
  return text.replace(INLINE_CITATION_RE, (match, idxStr: string) => {
    const idx = Number(idxStr);
    const url = citations[idx - 1];
    if (!url) return match; // unknown index — leave the literal marker
    return `[\\[${idx}\\]](${url})`;
  });
}

export interface SplitReasoningContent {
  content: string;
  reasoningContent: string;
}

/**
 * Convert model-emitted reasoning tags into the OpenAI-compatible
 * `reasoning_content` channel consumed by Agora and similar clients.
 */
export function splitReasoningContent(text: string): SplitReasoningContent {
  const escapedTagRegex = new RegExp(ESCAPED_REASONING_TAG_PATTERN, "gi");
  const reasoningTagRegex = new RegExp(REASONING_TAG_PATTERN, "gi");
  const normalized = text.replace(
    escapedTagRegex,
    (_match, closing: string, name: string) => `<${closing}${name.toLowerCase()}>`,
  );
  let content = "";
  let reasoningContent = "";
  let depth = 0;
  let cursor = 0;
  let sawOpeningTag = false;

  let match = reasoningTagRegex.exec(normalized);
  while (match !== null) {
    const preceding = normalized.slice(cursor, match.index);
    if (depth > 0) reasoningContent += preceding;
    else content += preceding;

    const closing = match[1] === "/";
    if (closing) {
      if (depth > 0) {
        depth--;
      } else if (!sawOpeningTag && reasoningContent.length === 0) {
        // DeepSeek-compatible endpoints sometimes omit the opening <think>
        // marker but still send </think>. In that dialect, everything before
        // the first closing marker is reasoning rather than answer content.
        reasoningContent = content;
        content = "";
      } else {
        content += match[0];
      }
    } else {
      sawOpeningTag = true;
      depth++;
    }
    cursor = match.index + match[0].length;
    match = reasoningTagRegex.exec(normalized);
  }

  const remainder = normalized.slice(cursor);
  if (depth > 0) reasoningContent += remainder;
  else content += remainder;
  return { content, reasoningContent };
}

/** Retained for callers that explicitly need inline-tag output. */
export function formatThinkTags(text: string): string {
  const split = splitReasoningContent(text);
  return split.reasoningContent
    ? `\n<think>\n${split.reasoningContent}\n</think>\n${split.content}`
    : split.content;
}

interface ChunkMeta {
  id: string;
  model: string;
  created: number;
  tools?: OpenAITool[];
  onChatId?: (chatId: string) => void;
}

/**
 * Wrap upstream text stream as OpenAI-format chat.completion.chunk SSE.
 *
 * Flush boundary: until either trailer marker (CITATIONS or CHAT_ID) appears
 * in the buffer, we hold back HOLDBACK_CHARS to guard against a partial
 * sentinel landing on a chunk boundary. Once a marker shows up, we lock the
 * flush boundary at the earliest marker and stop flushing past it, so the
 * trailer never reaches the client mid-stream.
 *
 * At stream end we parse the held-back tail, emit any remaining prose, and —
 * if perplexity citations were collected — append a Markdown sources block
 * as one final delta before the `[DONE]` terminator.
 */
export function streamUpstreamAsOpenAI(
  body: ReadableStream<Uint8Array>,
  meta: ChunkMeta,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  function sse(
    delta: ChatCompletionChunkDelta,
    finishReason: "stop" | null = null,
  ): Uint8Array {
    const chunk: ChatCompletionChunk = {
      id: meta.id,
      object: "chat.completion.chunk",
      created: meta.created,
      model: meta.model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    };
    return encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`);
  }

  /**
   * Emits a proper OpenAI-compatible tool-call delta chunk. Clients like
   * Agora watch for `delta.tool_calls[].function.{name,arguments}` and
   * `finish_reason: "tool_calls"` — they do not parse arbitrary JSON sitting
   * inside `delta.content`. Without this, a correctly-recognized
   * `relay_tool_call` payload still fails to trigger tool execution
   * downstream, even though the relay itself parsed it successfully.
   */
  function toolCallSse(toolCall: ShimToolCall): Uint8Array {
    const chunk: ChatCompletionChunk = {
      id: meta.id,
      object: "chat.completion.chunk",
      created: meta.created,
      model: meta.model,
      choices: [{
        index: 0,
        delta: {
          tool_calls: [{
            index: 0,
            id: `call_${meta.id}`,
            type: "function",
            function: {
                name: toolCall.function.name,
              // OpenAI schema requires arguments as a JSON *string*, not a
              // nested object — a common and separate failure point in
              // relay/harness bridges.
                arguments: toolCall.function.arguments,
            },
          }],
        },
        finish_reason: "tool_calls",
      }],
    };
    return encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`);
  }


  return new ReadableStream({
    async start(controller) {
      const reader = body.getReader();
      let pending = "";

      controller.enqueue(sse({ role: "assistant" }));

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value) continue;

          pending += decoder.decode(value, { stream: true });

          // Reasoning tags can span arbitrary upstream chunks. Keep the
          // response buffered so tags never leak into ordinary content.
        }

        pending += decoder.decode();
        const { content, citations, chatId } = parseTrailers(pending);
        if (chatId) meta.onChatId?.(chatId);
        const toolCall = tryParseRelayToolCall(content, meta.tools);
         if (toolCall) {
          controller.enqueue(toolCallSse(toolCall));
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
        return;
        }

        const split = splitReasoningContent(content);
        if (split.reasoningContent) {
          controller.enqueue(sse({ reasoning_content: split.reasoningContent }));
        }
        const visibleContent = split.content || split.reasoningContent;
        const answer = visibleContent + formatCitations(citations);
        if (answer) controller.enqueue(sse({ content: answer }));

        controller.enqueue(sse({}, "stop"));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });
}

interface ToolAwareChunkMeta {
  id: string;
  model: string;
  created: number;
  tools?: OpenAITool[];
  onChatId?: (chatId: string) => void;
}

/**
 * Tool-aware variant of streamUpstreamAsOpenAI. Buffers the entire response
 * (no incremental flush) so it can detect a full relay_tool_call JSON
 * envelope before deciding whether to emit tool_calls or plain content.
 * Only used when the request included `tools`.
 */
export function streamUpstreamWithToolShim(
  body: ReadableStream<Uint8Array>,
  meta: ToolAwareChunkMeta,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  function sse(
    delta: ChatCompletionChunkDelta,
    finishReason: "stop" | "tool_calls" | null = null,
  ): Uint8Array {
    const chunk: ChatCompletionChunk = {
      id: meta.id,
      object: "chat.completion.chunk",
      created: meta.created,
      model: meta.model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    };
    return encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`);
  }

  return new ReadableStream({
    async start(controller) {
      const reader = body.getReader();
      let buf = "";

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (value) buf += decoder.decode(value, { stream: true });
        }
        buf += decoder.decode();

        const { content, chatId, citations } = parseTrailers(buf);
        if (chatId) meta.onChatId?.(chatId);

        const toolCall = tryParseRelayToolCall(content, meta.tools);

        if (toolCall) {
          controller.enqueue(
            sse({
              role: "assistant",
              tool_calls: [
                {
                  index: 0,
                  id: toolCall.id,
                  type: "function",
                  function: toolCall.function,
                },
              ],
            }),
          );
          controller.enqueue(sse({}, "tool_calls"));
        } else {
          controller.enqueue(sse({ role: "assistant" }));
          const split = splitReasoningContent(content);
          if (split.reasoningContent) {
            controller.enqueue(sse({ reasoning_content: split.reasoningContent }));
          }
          const visibleContent = split.content || split.reasoningContent;
          const answer = visibleContent + formatCitations(citations);
          if (answer) controller.enqueue(sse({ content: answer }));
          controller.enqueue(sse({}, "stop"));
        }

        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });
}

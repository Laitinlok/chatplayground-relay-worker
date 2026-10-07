import type { OpenAIMessage } from "../types/openai";

export type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh";

const STARTERS = [
  "Let me restate the request and the constraints:",
  "Wait, let me check what that first pass missed:",
  "Let me challenge that and look for a better reading:",
  "Let me pull the verified points together:",
];

const TOOL_DECISION =
  'I should use xxx tool for xxx. Name the tool and the reason, or write "I should use no tool" when none applies:';

const TOOL_STARTERS = [
  TOOL_DECISION,
  "Wait, let me check whether a different tool must run first, or whether this call is unnecessary:",
  "Let me drop every tool that is not the immediate next step and keep only one call:",
  "The decision is one tool. Let me write that single TOOL_CALL and nothing else:",
];

/**
 * Luna does not emit a usable tool call unless it first drafts which tool
 * to use. Clients often omit reasoning_effort, so Luna defaults to medium.
 * An explicit "none" still disables the draft.
 */
export function resolveReasoningEffort(
  modelName: string,
  effort?: ReasoningEffort,
): ReasoningEffort | undefined {
  if (effort) return effort;
  return modelName.toLowerCase().includes("luna") ? "medium" : undefined;
}

/** Pass count for the draft-and-revise loop. 0 means a single normal call. */
export function reasoningPasses(effort?: ReasoningEffort): number {
  switch (effort) {
    case "low":
      return 1;
    case "medium":
      return 2;
    case "high":
      return 3;
    case "xhigh":
      return 4;
    default:
      return 0;
  }
}

/** Reasoning passes only. Tool choice is the XML shim, not a reasoning pass. */
export function prefillStarters(passes: number): string[] {
  return STARTERS.slice(0, Math.max(0, passes));
}

/**
 * Seed the assistant turn with the next sentence starter so the model
 * continues its own draft instead of answering immediately.
 */
export function withPrefill(
  messages: OpenAIMessage[],
  draft: string,
  starter: string,
): OpenAIMessage[] {
  const content = [draft.trim(), starter].filter(Boolean).join("\n\n");
  const seeded = [...messages];
  // Upstream rejects an assistant prefill, so continue the draft on the
  // last user turn. A later user message still sees it as prior text.
  const lastUserIndex = seeded.findLastIndex((message) => message.role === "user");
  const lastUser = lastUserIndex >= 0 ? seeded[lastUserIndex] : undefined;
  if (lastUser && typeof lastUser.content === "string") {
    seeded[lastUserIndex] = {
      ...lastUser,
      content: [lastUser.content.trim(), content].filter(Boolean).join("\n\n"),
    };
    return seeded.filter((message) => message.role !== "assistant" || message.tool_calls);
  }
  seeded.push({ role: "user", content });
  return seeded;
}

/** Keep only the text after the final answer marker, when the model emitted one. */
export function splitPrefillAnswer(draft: string): {
  reasoning: string;
  answer: string;
} {
  const marker = draft.lastIndexOf("**ANSWER**");
  if (marker < 0) return { reasoning: draft.trim(), answer: "" };
  return {
    reasoning: draft.slice(0, marker).trim(),
    answer: draft.slice(marker + "**ANSWER**".length).trim(),
  };
}

/**
 * Hide a tool-selection draft from the user-visible turn.
 * The draft names the next tool; the following model call must emit it.
 */
export function toolSelectionNote(draft: string): string {
  const notes = draft
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !/^TOOL_CALL\s*:/i.test(line) && !/^ARGUMENTS\s*:/i.test(line))
    .join("\n")
    .slice(0, 1200);
  if (!notes) return "";
  return [
    "<tool_selection>",
    notes,
    "</tool_selection>",
    "Emit only the single next TOOL_CALL named above. Do not answer yet and do not repeat this note.",
  ].join("\n");
}

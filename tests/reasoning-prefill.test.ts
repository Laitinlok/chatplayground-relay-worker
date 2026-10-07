import { describe, expect, it } from "vitest";
import {
  prefillStarters,
  reasoningPasses,
  resolveReasoningEffort,
  splitPrefillAnswer,
  toolSelectionNote,
  withPrefill,
} from "../src/utils/reasoning-prefill";

describe("reasoning prefill", () => {
  it("maps effort to a bounded pass count", () => {
    expect(reasoningPasses(undefined)).toBe(0);
    expect(reasoningPasses("none")).toBe(0);
    expect(reasoningPasses("low")).toBe(1);
    expect(reasoningPasses("medium")).toBe(2);
    expect(reasoningPasses("high")).toBe(3);
    expect(reasoningPasses("xhigh")).toBe(4);
    expect(prefillStarters(2)).toHaveLength(2);
    expect(prefillStarters(0)).toHaveLength(0);
    expect(prefillStarters(1)[0]).not.toContain("tool");
    expect(resolveReasoningEffort("gpt-6-luna", undefined)).toBe("medium");
    expect(resolveReasoningEffort("gpt-6-luna", "none")).toBe("none");
    expect(resolveReasoningEffort("sonar", undefined)).toBeUndefined();
  });

  it("continues an assistant draft instead of starting a new turn", () => {
    const seeded = withPrefill(
      [
        { role: "user", content: "Explain it" },
        { role: "assistant", content: "First pass" },
      ],
      "First pass",
      "Wait, let me check:",
    );
    expect(seeded).toHaveLength(1);
    expect(seeded[0]?.role).toBe("user");
    expect(seeded[0]?.content).toContain("First pass");
    expect(seeded[0]?.content).toContain("Wait, let me check:");
  });

  it("splits the final answer marker from the draft", () => {
    expect(splitPrefillAnswer("notes\n**ANSWER**\nfinal")).toEqual({
      reasoning: "notes",
      answer: "final",
    });
  });

  it("turns a tool draft into a hidden selection note", () => {
    const note = toolSelectionNote(
      'web_search is next because the claim is current.\nTOOL_CALL: web_search\nARGUMENTS: {"query":"claim"}',
    );
    expect(note).toContain("<tool_selection>");
    expect(note).toContain("web_search is next");
    expect(note).not.toContain("TOOL_CALL:");
    expect(note).toContain("Emit only the single next TOOL_CALL");
    expect(toolSelectionNote("   ")).toBe("");
  });
});

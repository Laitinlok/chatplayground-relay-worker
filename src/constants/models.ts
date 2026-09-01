// Seed registry — safety net used only if live discovery from
// chatplayground's /api/models feed fails. The live feed is authoritative.
//
// Note: upstreamBotId is NOT always the bare suffix of upstreamModel.
// Captured examples:
//   model="openai/gpt-5.5"               botId="gpt-5.5"
//   model="google/gemini-3-flash-preview" botId="gemini-3-flash"
// Always carry both fields independently.

import type { UpstreamEndpoint } from "./endpoints";

export interface ModelMetadata {
  context_window?: number;
  max_output_tokens?: number;
  supports_tools?: boolean;
  omniroute?: {
    task_fit: Record<string, number>;
    price_per_million_tokens: {
      input: number;
      output: number;
    };
  };
}

export interface ModelEntry extends ModelMetadata {
	id: string; // public id callers use (mirrors upstreamBotId)
	modelName: string; // bare model name, e.g. "gpt-5.5" / "sonar-pro"
	upstreamModel: string; // full slug, e.g. "google/gemini-3-flash-preview"
	upstreamBotId: string; // short id, e.g. "gemini-3-flash"
	provider: string; // "google"
	endpoint: UpstreamEndpoint; // which /api/chat/* endpoint serves this model
}

function m(
	provider: string,
	model: string,
	endpoint: UpstreamEndpoint = "azure",
	botId?: string,
): ModelEntry {
	const bot = botId ?? model;
	return applyModelProfile({
	  id: bot,
	  modelName: model,
	  upstreamModel: `${provider}/${model}`,
	  upstreamBotId: bot,
	  provider,
	  endpoint,
	});
}

function profile(
  context_window: number,
  max_output_tokens: number,
  supports_tools: boolean,
  task_fit: Record<string, number>,
  input: number,
  output: number,
): ModelMetadata {
  return {
    context_window,
    max_output_tokens,
    supports_tools,
    omniroute: {
      task_fit,
      price_per_million_tokens: { input, output },
    },
  };
}

export const MODEL_PROFILES: Record<string, ModelMetadata> = {
  "gpt-5.6-sol": profile(1_048_576, 32_768, true, { coding: 0.96, review: 0.94, planning: 0.91, analysis: 0.95, debugging: 0.93, documentation: 0.86, default: 0.92 }, 1.5, 6),
  "gpt-5.6-terra": profile(1_048_576, 32_768, true, { coding: 0.91, review: 0.93, planning: 0.96, analysis: 0.92, debugging: 0.88, documentation: 0.87, default: 0.91 }, 1.5, 6),
  "gpt-5.6-luna": profile(1_048_576, 32_768, true, { coding: 0.94, review: 0.9, planning: 0.89, analysis: 0.97, debugging: 0.95, documentation: 0.84, default: 0.91 }, 1.5, 6),
  "gemini-3.6-flash": profile(1_048_576, 16_384, true, { coding: 0.88, review: 0.84, planning: 0.86, analysis: 0.89, debugging: 0.82, documentation: 0.92, default: 0.86 }, 0.3, 1.2),
  "gemini-3-flash": profile(1_048_576, 16_384, true, { coding: 0.86, review: 0.82, planning: 0.83, analysis: 0.87, debugging: 0.8, documentation: 0.9, default: 0.84 }, 0.3, 1.2),
  "gemini-3.1-pro": profile(1_048_576, 32_768, true, { coding: 0.93, review: 0.9, planning: 0.92, analysis: 0.95, debugging: 0.89, documentation: 0.91, default: 0.92 }, 1.25, 5),
  "claude-fable-5": profile(200_000, 32_000, true, { coding: 0.9, review: 0.92, planning: 0.91, analysis: 0.93, debugging: 0.88, documentation: 0.94, default: 0.9 }, 3, 15),
  "claude-sonnet-5": profile(1_000_000, 32_000, true, { coding: 0.95, review: 0.95, planning: 0.94, analysis: 0.95, debugging: 0.92, documentation: 0.94, default: 0.94 }, 3, 15),
  "claude-opus-5": profile(1_000_000, 32_000, true, { coding: 0.97, review: 0.98, planning: 0.96, analysis: 0.98, debugging: 0.95, documentation: 0.9, default: 0.96 }, 5, 25),
  "claude-sonnet-4-6": profile(200_000, 16_384, true, { coding: 0.91, review: 0.92, planning: 0.9, analysis: 0.92, debugging: 0.89, documentation: 0.93, default: 0.91 }, 3, 15),
  "kimi-k3": profile(262_144, 16_384, true, { coding: 0.87, review: 0.84, planning: 0.86, analysis: 0.85, debugging: 0.83, documentation: 0.8, default: 0.84 }, 0.6, 2.5),
  "kimi-k2.6": profile(262_144, 16_384, true, { coding: 0.85, review: 0.82, planning: 0.84, analysis: 0.83, debugging: 0.8, documentation: 0.78, default: 0.82 }, 0.5, 2),
  "deepseek-v4-pro": profile(128_000, 32_768, true, { coding: 0.94, review: 0.9, planning: 0.86, analysis: 0.9, debugging: 0.94, documentation: 0.8, default: 0.89 }, 0.8, 3),
  "deepseek-v4-flash": profile(128_000, 16_384, true, { coding: 0.9, review: 0.84, planning: 0.8, analysis: 0.87, debugging: 0.9, documentation: 0.78, default: 0.85 }, 0.2, 0.8),
  "deepseek-r1": profile(64_000, 16_384, false, { coding: 0.88, review: 0.86, planning: 0.9, analysis: 0.93, debugging: 0.87, documentation: 0.74, default: 0.86 }, 0.55, 2.2),
  "glm-5.2": profile(128_000, 16_384, true, { coding: 0.84, review: 0.8, planning: 0.82, analysis: 0.84, debugging: 0.79, documentation: 0.83, default: 0.81 }, 0.4, 1.6),
  "llama-4-maverick": profile(1_000_000, 16_384, true, { coding: 0.82, review: 0.78, planning: 0.8, analysis: 0.81, debugging: 0.77, documentation: 0.79, default: 0.79 }, 0.2, 0.8),
  "minimax-m3": profile(1_000_000, 16_384, true, { coding: 0.86, review: 0.82, planning: 0.87, analysis: 0.84, debugging: 0.8, documentation: 0.86, default: 0.84 }, 0.3, 1.2),
  "llama-4-scout": profile(512_000, 16_384, true, { coding: 0.8, review: 0.76, planning: 0.78, analysis: 0.79, debugging: 0.74, documentation: 0.77, default: 0.76 }, 0.15, 0.6),
  "hy3": profile(128_000, 16_384, true, { coding: 0.78, review: 0.74, planning: 0.8, analysis: 0.77, debugging: 0.72, documentation: 0.76, default: 0.75 }, 0.3, 1.2),
  "mimo-v2.5-pro": profile(128_000, 16_384, true, { coding: 0.88, review: 0.83, planning: 0.84, analysis: 0.86, debugging: 0.85, documentation: 0.8, default: 0.84 }, 0.4, 1.6),
  "nova-2-lite-v1": profile(300_000, 16_384, true, { coding: 0.76, review: 0.72, planning: 0.75, analysis: 0.74, debugging: 0.7, documentation: 0.82, default: 0.73 }, 0.2, 0.8),
  "command-a": profile(256_000, 16_384, true, { coding: 0.83, review: 0.8, planning: 0.82, analysis: 0.81, debugging: 0.77, documentation: 0.9, default: 0.81 }, 2.5, 10),
  "qwen3.8-max": profile(256_000, 32_768, true, { coding: 0.92, review: 0.87, planning: 0.88, analysis: 0.9, debugging: 0.9, documentation: 0.84, default: 0.88 }, 0.8, 3.2),
  "qwen3.7-plus": profile(256_000, 16_384, true, { coding: 0.88, review: 0.83, planning: 0.84, analysis: 0.86, debugging: 0.85, documentation: 0.81, default: 0.84 }, 0.5, 2),
  "qwen3.6-flash": profile(128_000, 16_384, true, { coding: 0.83, review: 0.78, planning: 0.8, analysis: 0.81, debugging: 0.79, documentation: 0.79, default: 0.79 }, 0.25, 1),
  "grok-4.5": profile(256_000, 32_768, true, { coding: 0.9, review: 0.86, planning: 0.89, analysis: 0.88, debugging: 0.84, documentation: 0.8, default: 0.86 }, 3, 15),
  "perplexity-sonar-reasoning-pro": profile(200_000, 16_384, false, { coding: 0.7, review: 0.74, planning: 0.73, analysis: 0.91, debugging: 0.68, documentation: 0.84, default: 0.79 }, 3, 15),
  "perplexity-sonar-pro": profile(200_000, 8_192, false, { coding: 0.68, review: 0.7, planning: 0.66, analysis: 0.8, debugging: 0.64, documentation: 0.78, default: 0.72 }, 3, 15),
  "perplexity-sonar": profile(128_000, 8_192, false, { coding: 0.62, review: 0.66, planning: 0.64, analysis: 0.77, debugging: 0.58, documentation: 0.75, default: 0.68 }, 1, 5),
  "mistral-large-3": profile(256_000, 32_768, true, { coding: 0.86, review: 0.82, planning: 0.84, analysis: 0.85, debugging: 0.81, documentation: 0.87, default: 0.83 }, 2, 6),
  "command-r-plus": profile(128_000, 16_384, true, { coding: 0.8, review: 0.78, planning: 0.8, analysis: 0.79, debugging: 0.74, documentation: 0.88, default: 0.79 }, 2.5, 10),
};

export function applyModelProfile(model: ModelEntry): ModelEntry {
  return { ...model, ...(MODEL_PROFILES[model.id] ?? {}) };
}

export const SEED_MODELS: ModelEntry[] = [
  m("openai", "gpt-5.6-sol"),
  m("openai", "gpt-5.6-terra"),
  m("openai", "gpt-5.6-luna"),
  m("google", "gemini-3.6-flash"),
  m("google", "gemini-3-flash"),
  m("google", "gemini-3.1-pro"),
  m("anthropic", "claude-fable-5"),
  m("anthropic", "claude-sonnet-5"),
  m("anthropic", "claude-opus-5"),
  m("anthropic", "claude-sonnet-4-6"),
  m("kimi", "kimi-k3"),
  m("kimi", "kimi-k2.6"),
  m("deepseek", "deepseek-v4-pro"),
  m("deepseek", "deepseek-v4-flash"),
  m("deepseek", "deepseek-r1"),
  m("zai", "glm-5.2"),
  m("meta", "llama-4-maverick"),
  m("minimax", "minimax-m3"),
  m("meta", "llama-4-scout"),
  m("tencent", "hy3"),
  m("xiaomi", "mimo-v2.5-pro"),
  m("amazon", "nova-2-lite-v1"),
  m("cohere", "command-a"),
  m("qwen", "qwen3.8-max"),
  m("qwen", "qwen3.7-plus"),
  m("qwen", "qwen3.6-flash"),
  m("xai", "grok-4.5"),
  m("perplexity", "perplexity-sonar-reasoning-pro", "perplexity"),
  m("perplexity", "perplexity-sonar-pro", "perplexity"),
  m("perplexity", "perplexity-sonar", "perplexity"),
  m("mistral", "mistral-large-3"),
  m("cohere", "command-r-plus"),
];

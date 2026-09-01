import type { OpenAIContentPart, OpenAIMessage } from "../types/openai";
import type { ChatCompletionRequest } from "../types/openai";
import type { OpenAITool, ToolChoice } from "../utils/tool-shim";

export interface ResponsesRequest {
	model: string;
	input: string | ResponsesInputItem[];
	instructions?: string;
	previous_response_id?: string;
	stream?: boolean;
	user?: string;
	tools?: ResponsesTool[];
	tool_choice?: ResponsesToolChoice;
	metadata?: Record<string, string>;
	temperature?: number;
	top_p?: number;
	max_output_tokens?: number;
	reasoning?: { effort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh"; summary?: "auto" | "concise" | "detailed" };
	parallel_tool_calls?: boolean;
	store?: boolean;
}

export type ResponsesTool =
	| {
			type: "function";
			name: string;
			description?: string;
			parameters?: unknown;
			strict?: boolean;
	  }
	| { type: "function"; function: OpenAITool["function"] };

export type ResponsesToolChoice =
	| "none"
	| "auto"
	| "required"
	| { type: "function"; name?: string; function?: { name: string } };

export type ResponsesInputItem = {
	type?: string;
	role?: "developer" | "system" | "user" | "assistant" | "tool";
	content?: unknown;
	name?: string;
	call_id?: string;
	id?: string;
	output?: unknown;
	arguments?: string;
};

export interface ResponsesOutputText {
	type: "output_text";
	text: string;
	annotations: unknown[];
}

export interface ResponsesMessageOutput {
	type: "message";
	id: string;
	role: "assistant";
	status: "completed";
	content: ResponsesOutputText[];
}

export interface ResponsesFunctionCallOutput {
	type: "function_call";
	id: string;
	call_id: string;
	name: string;
	arguments: string;
	status: "completed";
}

export type ResponsesOutput =
	| ResponsesMessageOutput
	| ResponsesFunctionCallOutput;

export interface ResponsesUsage {
	input_tokens: number;
	output_tokens: number;
	total_tokens: number;
}

export interface ResponsesResponse {
	id: string;
	object: "response";
	created_at: number;
	status: "completed";
	model: string;
	output: ResponsesOutput[];
	output_text: string;
	reasoning_content?: string;
	usage: ResponsesUsage;
}

function contentParts(value: unknown): string | OpenAIContentPart[] {
	if (typeof value === "string") return value;
	if (!Array.isArray(value)) return "";
	const parts: OpenAIContentPart[] = [];
	for (const part of value) {
		if (!part || typeof part !== "object") continue;
		const p = part as Record<string, unknown>;
		if (p.type === "input_text" || p.type === "text") {
			if (typeof p.text === "string")
				parts.push({ type: "text", text: p.text });
		} else if (p.type === "input_image" || p.type === "image_url") {
			const url =
				typeof p.image_url === "string"
					? p.image_url
					: p.image_url &&
							typeof p.image_url === "object" &&
							typeof (p.image_url as Record<string, unknown>).url === "string"
						? (p.image_url as Record<string, string>).url
						: typeof p.url === "string"
							? p.url
							: null;
			if (url) parts.push({ type: "image_url", image_url: { url } });
		}
	}
	return parts;
}

export function responsesToolsToChatTools(
	tools?: ResponsesTool[],
): OpenAITool[] | undefined {
	if (!tools) return undefined;
	return tools.map((tool) => {
		if ("function" in tool)
			return { type: "function", function: tool.function };
		return {
			type: "function",
			function: {
				name: tool.name,
				description: tool.description,
				parameters: tool.parameters,
			},
		};
	});
}

export function responsesToolChoiceToChatChoice(
	choice?: ResponsesToolChoice,
): ToolChoice | undefined {
	if (choice === "required") return "required";
	if (choice && typeof choice === "object") {
		const name = choice.name ?? choice.function?.name;
		return name ? { type: "function", function: { name } } : "auto";
	}
	return choice;
}

export function responsesToChatRequest(
	req: ResponsesRequest,
): ChatCompletionRequest {
	const items: ResponsesInputItem[] =
	  typeof req.input === "string"
	    ? [{ role: "user", content: req.input }]
	    : req.input;
	const messages: OpenAIMessage[] = [];
	if (req.instructions) messages.push({ role: "system", content: req.instructions });
	for (const item of items) {
		if (item.type === "function_call") {
			messages.push({
				role: "assistant",
				content: null,
				tool_calls: [
					{
						id: item.call_id ?? item.id ?? `call_${crypto.randomUUID()}`,
						type: "function",
						function: {
							name: item.name ?? "",
							arguments: item.arguments ?? "{}",
						},
					},
				],
			});
			continue;
		}
		if (item.type === "function_call_output") {
			const output =
				typeof item.output === "string"
					? item.output
					: JSON.stringify(item.output ?? "");
			messages.push({ role: "tool", content: output, name: item.call_id });
			continue;
		}
		const role = item.role === "developer" ? "system" : item.role;
		if (!role || !["system", "user", "assistant", "tool"].includes(role))
			continue;
		messages.push({
			role: role as OpenAIMessage["role"],
			content: contentParts(item.content),
		});
	}
	return {
		model: req.model,
		messages,
		stream: req.stream,
		user: req.user,
		tools: responsesToolsToChatTools(req.tools),
		tool_choice: responsesToolChoiceToChatChoice(req.tool_choice),
		metadata: req.metadata ? { save: false } : undefined,
		max_tokens: req.max_output_tokens,
		temperature: req.temperature,
		top_p: req.top_p,
		reasoning_effort: req.reasoning?.effort,
  };
}

export function chatResultToResponses(
	model: string,
	content: string,
	toolCall: {
		id: string;
		type: "function";
		function: { name: string; arguments: string };
	} | null,
	inputMessages: OpenAIMessage[],
	id = `resp_${crypto.randomUUID().replace(/-/g, "")}`,
	reasoningContent = "",
): ResponsesResponse {
	const created_at = Math.floor(Date.now() / 1000);
	const output: ResponsesOutput[] = toolCall
		? [
				{
					type: "function_call",
					id: `fc_${toolCall.id.replace(/^call_/, "")}`,
					call_id: toolCall.id,
					name: toolCall.function.name,
					arguments: toolCall.function.arguments,
					status: "completed",
				},
			]
		: [
				{
					type: "message",
					id: `msg_${id.replace(/^resp_/, "")}`,
					role: "assistant",
					status: "completed",
					content: [{ type: "output_text", text: content, annotations: [] }],
				},
			];
	const input_tokens = Math.ceil(
		inputMessages.reduce(
			(sum, message) =>
				sum +
				(typeof message.content === "string" ? message.content.length : 0),
			0,
		) / 4,
	);
	const output_tokens = Math.ceil(content.length / 4);
	return {
		id,
		object: "response",
		created_at,
		status: "completed",
		model,
		output,
		output_text: toolCall ? "" : content,
		...(reasoningContent ? { reasoning_content: reasoningContent } : {}),
		usage: {
			input_tokens,
			output_tokens,
			total_tokens: input_tokens + output_tokens,
		},
	};
}

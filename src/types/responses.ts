import type { OpenAIContentPart, OpenAIMessage } from "../types/openai";
import type { ChatCompletionRequest } from "../types/openai";
import type { OpenAITool, ToolChoice } from "../utils/tool-shim";

export interface ResponsesRequest {
  model: string;
  input?: string | ResponsesInputItem | ResponsesInputItem[];
  // Compatibility fields used by clients that send Chat Completions-shaped
  // history to a Responses-compatible endpoint.
  messages?: ResponsesInputItem[];
  prompt?: string;
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
  reasoning?: {
    effort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh";
    summary?: "auto" | "concise" | "detailed";
  };
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
  | {
      type: "web_search" | "web_search_preview";
      search_context_size?: "low" | "medium" | "high";
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
  annotations: Array<{
    type: "url_citation";
    url: string;
    title?: string;
    start_index?: number;
    end_index?: number;
  }>;
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

export interface ResponsesWebSearchResult {
  title: string;
  url: string;
  snippet: string;
}

export interface ResponsesWebSearchCallOutput {
  type: "web_search_call";
  id: string;
  status: "completed";
  action: {
    type: "search";
    query: string;
    results?: ResponsesWebSearchResult[];
  };
}

export type ResponsesOutput =
  | ResponsesMessageOutput
  | ResponsesFunctionCallOutput
  | ResponsesWebSearchCallOutput;

export interface ResponsesUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface ResponsesResponse {
  id: string;
  object: "response";
  created_at: number;
  status: "in_progress" | "completed" | "failed";
  model: string;
  output: ResponsesOutput[];
  output_text: string;
  reasoning_content?: string;
  error?: {
    code?: string;
    message: string;
    type?: string;
  };
  usage: ResponsesUsage;
}

function extractSearchQuery(argumentsJson: string): string {
  try {
    const parsed: unknown = JSON.parse(argumentsJson);
    if (!parsed || typeof parsed !== "object") return "";
    const value = parsed as { query?: unknown; queries?: unknown };
    if (typeof value.query === "string") return value.query;
    if (Array.isArray(value.queries))
      return (
        value.queries.find(
          (query): query is string => typeof query === "string",
        ) ?? ""
      );
  } catch {
    // Keep the output valid even when the model emitted malformed arguments.
  }
  return "";
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
    if (tool.type === "web_search" || tool.type === "web_search_preview") {
      return {
        type: "function",
        function: {
          name: "web_search",
          description: "Search the web for current information.",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string" },
              num_results: {
                type: "integer",
                description:
                  "Optional number of search results needed to answer the request. Choose the smallest useful number.",
                minimum: 1,
                maximum: 10,
              },
              queries: { type: "array", items: { type: "string" } },
            },
            additionalProperties: false,
          },
        },
      };
    }
    if (tool.type === "function") {
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
    }
    throw new Error("Unsupported Responses tool type");
  });
}

export function hasNativeWebSearch(tools?: ResponsesTool[]): boolean {
  return Boolean(
    tools?.some(
      (tool) =>
        tool.type === "web_search" || tool.type === "web_search_preview",
    ),
  );
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
  const source = req.input ?? req.messages ?? req.prompt ?? [];
  const items: ResponsesInputItem[] =
    typeof source === "string"
      ? [{ role: "user", content: source }]
      : Array.isArray(source)
        ? source
        : [source];
  const messages: OpenAIMessage[] = [];
  if (req.instructions)
    messages.push({ role: "system", content: req.instructions });
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
  nativeWebSearch = false,
  searchResults: ResponsesWebSearchResult[] = [],
): ResponsesResponse {
  const created_at = Math.floor(Date.now() / 1000);
  const searchQuery = toolCall
    ? extractSearchQuery(toolCall.function.arguments)
    : "";
  const searchOutput: ResponsesWebSearchCallOutput | null =
    toolCall && nativeWebSearch && toolCall.function.name === "web_search"
      ? {
          type: "web_search_call",
          id: `ws_${toolCall.id.replace(/^call_/, "")}`,
          status: "completed",
          action: {
            type: "search",
            query: searchQuery,
            ...(searchResults.length > 0 ? { results: searchResults } : {}),
          },
        }
      : null;
  const annotations = searchResults.flatMap((result) => {
    const start = content.indexOf(result.title);
    return [
      {
        type: "url_citation" as const,
        url: result.url,
        title: result.title,
        ...(start >= 0
          ? { start_index: start, end_index: start + result.title.length }
          : {}),
      },
    ];
  });
  const messageOutput: ResponsesMessageOutput = {
    type: "message",
    id: `msg_${id.replace(/^resp_/, "")}`,
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text: content, annotations }],
  };
  const output: ResponsesOutput[] = toolCall
    ? searchOutput
      ? [searchOutput, ...(content ? [messageOutput] : [])]
      : [
          {
            type: "function_call",
            id: `fc_${toolCall.id.replace(/^call_/, "")}`,
            call_id: toolCall.id,
            name: toolCall.function.name,
            arguments: toolCall.function.arguments,
            status: "completed",
          },
        ]
    : [messageOutput];
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
    output_text: toolCall && !searchOutput ? "" : content,
    ...(reasoningContent ? { reasoning_content: reasoningContent } : {}),
    usage: {
      input_tokens,
      output_tokens,
      total_tokens: input_tokens + output_tokens,
    },
  };
}

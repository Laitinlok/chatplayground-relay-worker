const savedRelayUrl = localStorage.getItem("relayUrl") || sessionStorage.getItem("relayUrl") || "";
const savedRelayKey = localStorage.getItem("relayKey") || sessionStorage.getItem("relayKey") || "";
const savedMcpServers = localStorage.getItem("mcpServers") || sessionStorage.getItem("mcpServers") || "[]";
const savedSelectedModels = JSON.parse(localStorage.getItem("selectedModels") || "[]");
const savedSynthesisModel = localStorage.getItem("synthesisModel") || "";
const savedConversations = JSON.parse(localStorage.getItem("conversations") || "[]");
const initialConversation = savedConversations[0] || { id: crypto.randomUUID(), title: "New conversation", messages: [], request: null, lastComparison: null, lastSynthesis: "" };

const state = {
  relayUrl: savedRelayUrl,
  relayKey: savedRelayKey,
  mcpServers: JSON.parse(savedMcpServers),
  mcpTools: [],
  models: [],
  conversations: savedConversations.length ? savedConversations : [initialConversation],
  activeConversationId: initialConversation.id,
  messages: initialConversation.messages || [],
  selected: savedSelectedModels,
  synthesisModel: savedSynthesisModel,
  request: initialConversation.request || null,
  lastComparison: initialConversation.lastComparison || null,
  lastSynthesis: initialConversation.lastSynthesis || "",
  replyContext: "all",
  activeViewpointIndex: null,
  theme: localStorage.getItem("theme") || "light",
};

const $ = (selector) => document.querySelector(selector);
const els = {
  sidebar: $("#sidebar"),
  welcome: $("#welcome"),
  conversation: $("#conversation"),
  prompt: $("#prompt"),
  modelList: $("#modelList"),
  relayUrl: $("#relayUrl"),
  relayKey: $("#relayKey"),
  synthesisModel: $("#synthesisModel"),
  mcpServers: $("#mcpServers"),
  mcpStatus: $("#mcpStatus"),
  dialog: $("#settingsDialog"),
  statusDot: $("#statusDot"),
  connectionLabel: $("#connectionLabel"),
  connectionDetail: $("#connectionDetail"),
  themeToggle: $("#themeToggle"),
  toggleViewpoint: $("#toggleViewpoint"),
  resultList: $("#resultList"),
  historyList: $("#historyList"),
  historyCount: $("#historyCount"),
  viewpointPanel: $("#viewpointPanel"),
  viewpointTitle: $("#viewpointTitle"),
  viewpointContent: $("#viewpointContent"),
  replyViewpoint: $("#replyViewpoint"),
};

function normalizeRelayUrl(value) {
  return value.trim().replace(/\/+$/, "").replace(/\/v1$/i, "");
}

function apiUrl(path) {
  return `${state.relayUrl}${path}`;
}

function headers() {
  return { "Content-Type": "application/json", Authorization: `Bearer ${state.relayKey}` };
}

function setConnected(connected, detail = "") {
  els.statusDot.classList.toggle("connected", connected);
  els.connectionLabel.textContent = connected ? "Relay connected" : "Not connected";
  els.connectionDetail.textContent = connected ? detail : (detail || "Add your relay endpoint");
}

function openSettings() {
  els.relayUrl.value = state.relayUrl;
  els.relayKey.value = state.relayKey;
  els.mcpServers.value = JSON.stringify(state.mcpServers, null, 2);
  els.synthesisModel.value = state.synthesisModel;
  els.dialog.showModal();
}

async function saveSettings(event) {
  event.preventDefault();
  state.relayUrl = normalizeRelayUrl(els.relayUrl.value);
  state.relayKey = els.relayKey.value.trim();
  state.synthesisModel = els.synthesisModel.value;
  try {
    const configured = JSON.parse(els.mcpServers.value || "[]");
    if (!Array.isArray(configured) || configured.some((server) => !server.name || !server.url)) throw new Error("MCP servers must be an array of { name, url } objects.");
    state.mcpServers = configured.map((server) => ({ name: String(server.name), url: String(server.url), headers: server.headers || {} }));
  } catch (error) {
    els.mcpStatus.textContent = error.message;
    return;
  }
  localStorage.setItem("relayUrl", state.relayUrl);
  localStorage.setItem("relayKey", state.relayKey);
  localStorage.setItem("mcpServers", JSON.stringify(state.mcpServers));
  localStorage.setItem("synthesisModel", state.synthesisModel);
  localStorage.setItem("selectedModels", JSON.stringify(state.selected));
  sessionStorage.setItem("relayUrl", state.relayUrl);
  sessionStorage.setItem("relayKey", state.relayKey);
  sessionStorage.setItem("mcpServers", JSON.stringify(state.mcpServers));
  els.dialog.close();
  await discoverMcpTools();
  await loadModels();
}

const embeddedDuckDuckGoServer = { name: "DuckDuckGo MCP (embedded)", url: "embedded://duckduckgo", headers: {}, embedded: true };

async function mcpRequest(server, method, params = {}, sessionId) {
  if (server.embedded) {
    if (method === "initialize") return { result: { protocolVersion: "2025-03-26", capabilities: { tools: {} }, serverInfo: { name: server.name, version: "embedded" } }, sessionId: "embedded-duckduckgo" };
    if (method === "tools/list") return { result: { tools: [{ name: "duckduckgo_search", description: "Search the web with DuckDuckGo.", inputSchema: { type: "object", properties: { query: { type: "string" }, max_results: { type: "integer" } }, required: ["query"] } }] }, sessionId };
    if (method === "tools/call" && params.name === "duckduckgo_search") {
      const invoke = window.__TAURI__?.core?.invoke;
      if (!invoke) throw new Error("The native DuckDuckGo MCP bridge is unavailable. Run the Tauri AppImage.");
      const text = await invoke("duckduckgo_search", { query: params.arguments?.query || "", maxResults: params.arguments?.max_results || 10, region: params.arguments?.region || "" });
      return { result: { content: [{ type: "text", text }] }, sessionId };
    }
  }
  const response = await fetch(server.url, { method: "POST", headers: { Accept: "application/json, text/event-stream", "Content-Type": "application/json", ...(server.headers || {}), ...(sessionId ? { "Mcp-Session-Id": sessionId } : {}) }, body: JSON.stringify({ jsonrpc: "2.0", id: crypto.randomUUID(), method, params }) });
  if (!response.ok) throw new Error(`${server.name}: MCP request failed (${response.status})`);
  const raw = await response.text();
  let payload;
  try { payload = JSON.parse(raw); }
  catch { payload = raw.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trim()).reverse().map((line) => { try { return JSON.parse(line); } catch { return null; } }).find(Boolean); }
  if (!payload) throw new Error(`${server.name}: MCP returned an unreadable response.`);
  if (payload.error) throw new Error(`${server.name}: ${payload.error.message || "MCP error"}`);
  return { result: payload.result || {}, sessionId: response.headers.get("Mcp-Session-Id") || sessionId };
}

async function discoverMcpTools() {
  const builtInTool = { server: embeddedDuckDuckGoServer, sessionId: "embedded-duckduckgo", tool: { name: "duckduckgo_search", description: "Search the web with DuckDuckGo.", inputSchema: { type: "object", properties: { query: { type: "string" }, max_results: { type: "integer" } }, required: ["query"] } } };
  state.mcpTools = window.__TAURI__?.core?.invoke ? [builtInTool] : [];
  if (!state.mcpServers.length) {
    els.mcpStatus.textContent = state.mcpTools.length
      ? "Embedded DuckDuckGo MCP ready."
      : "Add an MCP server in Settings to enable tools.";
    return;
  }
  try {
    for (const server of state.mcpServers) {
      const initialized = await mcpRequest(server, "initialize", { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "relay-studio", version: "0.1.0" } });
      const listed = await mcpRequest(server, "tools/list", {}, initialized.sessionId);
      for (const tool of listed.result.tools || []) state.mcpTools.push({ server, sessionId: listed.sessionId, tool });
    }
    els.mcpStatus.textContent = `${state.mcpTools.length} MCP tools discovered across ${state.mcpServers.length} server${state.mcpServers.length === 1 ? "" : "s"}.`;
  } catch (error) { els.mcpStatus.textContent = error.message; }
}

async function executeMcpTool(name, args) {
  const entry = state.mcpTools.find((item) => item.tool.name === name);
  if (!entry) throw new Error(`MCP tool not found: ${name}`);
  const response = await mcpRequest(entry.server, "tools/call", { name: entry.tool.name, arguments: args }, entry.sessionId);
  return response.result;
}
async function loadModels() {
  if (!state.relayUrl || !state.relayKey) {
    renderModels();
    setConnected(false);
    return;
  }
  els.modelList.innerHTML = '<span class="muted">Loading available models...</span>';
  try {
    const response = await fetch(apiUrl("/v1/models"), { headers: { Authorization: `Bearer ${state.relayKey}` } });
    if (!response.ok) throw new Error(`Model discovery failed (${response.status})`);
    const payload = await response.json();
    state.models = Array.isArray(payload.data) ? payload.data : [];
    state.selected = state.selected.filter((id) => state.models.some((model) => model.id === id));
    if (!state.selected.length) state.selected = state.models.slice(0, Math.min(3, state.models.length)).map((model) => model.id);
    localStorage.setItem("selectedModels", JSON.stringify(state.selected));
    renderModels();
    renderSynthesisModels();
    setConnected(true, `${state.models.length} models available`);
  } catch (error) {
    state.models = [];
    renderModels();
    setConnected(false, error instanceof Error ? error.message : String(error));
  }
}

function renderModels() {
  if (!state.models.length) {
    els.modelList.innerHTML = '<span class="muted">Connect a relay to load models.</span>';
    return;
  }
  els.modelList.innerHTML = state.models.map((model) => `
    <label class="model-option">
      <input type="checkbox" value="${escapeHtml(model.id)}" ${state.selected.includes(model.id) ? "checked" : ""}>
      <span>${escapeHtml(model.id)}</span>
    </label>`).join("");
  els.modelList.querySelectorAll("input").forEach((input) => input.addEventListener("change", (event) => {
    const id = event.target.value;
    state.selected = event.target.checked ? [...state.selected, id] : state.selected.filter((item) => item !== id);
    localStorage.setItem("selectedModels", JSON.stringify(state.selected));
    renderSynthesisModels();
  }));
}

function renderSynthesisModels() {
  els.synthesisModel.innerHTML = '<option value="">Use the first selected model</option>' + state.models.map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.id)}</option>`).join("");
  if (state.models.some((model) => model.id === state.synthesisModel)) els.synthesisModel.value = state.synthesisModel;
}

function toggle(button) {
  const active = button.getAttribute("aria-pressed") === "true";
  button.setAttribute("aria-pressed", String(!active));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  })[character]);
}

function renderText(value) {
  let html = escapeHtml(value);
  html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  html = html.replace(/^### (.*)$/gm, '<h4>$1</h4>').replace(/^## (.*)$/gm, '<h3>$1</h3>').replace(/^# (.*)$/gm, '<h2>$1</h2>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  html = html.replace(/^(?:- |• )(.*)$/gm, '<li>$1</li>').replace(/(?:<li>.*<\/li>\n?)+/g, (list) => `<ul>${list}</ul>`);
  return html;
}

function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
  if (els.themeToggle) els.themeToggle.textContent = state.theme === "dark" ? "Light" : "Dark";
}

function synthesizedResults() {
  return state.messages.map((message, index) => ({ message, index })).filter(({ message }) => message.kind === "assistant" && message.label && !message.error && /^(Combine|Comparison)/.test(message.label));
}

function setViewpointVisibility(visible) {
  els.viewpointPanel.classList.toggle("open", visible);
  els.toggleViewpoint.setAttribute("aria-pressed", String(visible));
  els.toggleViewpoint.textContent = visible ? "Hide viewpoint" : "Viewpoint";
}

function selectViewpoint(index) {
  const result = state.messages[index];
  if (!result || result.kind !== "assistant") return;
  state.activeViewpointIndex = index;
  setViewpointVisibility(true);
  els.viewpointTitle.textContent = result.label;
  els.viewpointContent.innerHTML = renderText(result.content);
  els.replyViewpoint.disabled = false;
  renderSidebarResults();
}

function persistConversation() {
  const conversation = state.conversations.find((item) => item.id === state.activeConversationId);
  if (!conversation) return;
  conversation.messages = state.messages;
  conversation.request = state.request;
  conversation.lastComparison = state.lastComparison;
  conversation.lastSynthesis = state.lastSynthesis;
  const firstPrompt = state.messages.find((message) => message.kind === "user")?.content;
  if (firstPrompt) conversation.title = firstPrompt.slice(0, 60);
  localStorage.setItem("conversations", JSON.stringify(state.conversations));
}

function loadConversation(id) {
  persistConversation();
  const conversation = state.conversations.find((item) => item.id === id);
  if (!conversation) return;
  state.activeConversationId = id;
  state.messages = conversation.messages || [];
  state.request = conversation.request || null;
  state.lastComparison = conversation.lastComparison || null;
  state.lastSynthesis = conversation.lastSynthesis || "";
  state.activeViewpointIndex = null;
  setViewpointVisibility(false);
  renderConversation();
}

function createConversation() {
  persistConversation();
  const conversation = { id: crypto.randomUUID(), title: "New conversation", messages: [], request: null, lastComparison: null, lastSynthesis: "" };
  state.conversations.unshift(conversation);
  state.activeConversationId = conversation.id;
  state.messages = conversation.messages;
  state.request = null;
  state.lastComparison = null;
  state.lastSynthesis = "";
  state.replyContext = "all";
  state.activeViewpointIndex = null;
  els.prompt.value = "";
  els.prompt.placeholder = "Ask anything across your selected models...";
  setViewpointVisibility(false);
  persistConversation();
  renderConversation();
}

function deleteConversation(id) {
  const deletingActive = id === state.activeConversationId;
  state.conversations = state.conversations.filter((conversation) => conversation.id !== id);
  if (!state.conversations.length) {
    const replacement = { id: crypto.randomUUID(), title: "New conversation", messages: [], request: null, lastComparison: null, lastSynthesis: "" };
    state.conversations.push(replacement);
  }
  localStorage.setItem("conversations", JSON.stringify(state.conversations));
  if (deletingActive) loadConversation(state.conversations[0].id);
  else renderSidebarResults();
}

function renderSidebarResults() {
  els.historyCount.textContent = String(state.conversations.length);
  els.historyList.innerHTML = state.conversations.map((conversation) => `<div class="history-row ${conversation.id === state.activeConversationId ? "active" : ""}"><button class="history-item" data-conversation-id="${conversation.id}" type="button">${escapeHtml(conversation.title || "New conversation")}</button><button class="delete-history" data-delete-conversation="${conversation.id}" type="button" aria-label="Delete ${escapeHtml(conversation.title || "conversation")}" title="Delete conversation">Delete</button></div>`).join("");

  const results = synthesizedResults();
  els.resultList.innerHTML = results.length ? results.slice().reverse().map(({ message, index }) => `<button class="viewpoint-item ${index === state.activeViewpointIndex ? "active" : ""}" data-result-index="${index}" type="button"><strong>${escapeHtml(message.label)}</strong></button>`).join("") : "";
}

function renderConversation() {
  els.welcome.hidden = state.messages.length > 0;
  els.conversation.hidden = false;
  els.conversation.innerHTML = state.messages.filter((message) => !(message.kind === "assistant" && message.label && /^(Combine|Comparison)/.test(message.label))).map((message) => {
    if (message.kind === "compare") {
      return `<div class="compare-grid">${message.items.map((item) => `<article class="compare-card"><div class="message-label"><span>${escapeHtml(item.model)}</span><span>${item.error ? "Error" : "Response"}</span></div>${item.searchQuery ? `<div class="model-search-query">Searched: ${escapeHtml(item.searchQuery)}</div>` : ""}<div class="message-body ${item.error ? "error" : ""}">${renderText(item.content)}</div>${item.sources?.length ? `<div class="sources">${item.sources.map((source) => `<a class="source" href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.title || source.url)}</a>`).join("")}</div>` : ""}</article>`).join("")}</div>`;
    }
    const sources = message.sources?.length ? `<div class="sources">${message.sources.map((source) => `<a class="source" href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.title || source.url)}</a>`).join("")}</div>` : "";
    const label = message.kind === "user" ? "You" : message.kind === "search" ? "Web search" : message.label || "Response";
    return `<article class="message ${message.kind === "user" ? "user" : ""}"><div class="message-label"><span>${label}</span><span>${message.label || ""}</span></div><div class="message-body ${message.error ? "error" : ""}">${renderText(message.content)}</div>${sources}</article>`;
  }).join("");
  persistConversation();
  renderSidebarResults();
  els.conversation.scrollTop = els.conversation.scrollHeight;
}

async function duckDuckGoMcpSearch(query) {
  const candidates = state.mcpTools.filter(({ server, tool }) => server.embedded || /duckduckgo|duck duck go/i.test(server.name) || /duckduckgo/i.test(tool.name));
  const entry = candidates.find(({ tool }) => /search/i.test(tool.name)) || candidates[0];
  if (!entry) throw new Error("DuckDuckGo MCP is not configured. Add a DuckDuckGo MCP server in Settings and save it.");

  const properties = entry.tool.inputSchema?.properties || {};
  const args = {};
  if (!Object.keys(properties).length || properties.query) args.query = query;
  if (properties.max_results) args.max_results = 6;
  else if (properties.num_results) args.num_results = 6;
  else if (properties.count) args.count = 6;
  const result = await executeMcpTool(entry.tool.name, args);
  const content = result?.content || [];
  const text = content.filter((item) => item?.type === "text").map((item) => item.text).join("\n\n");
  const structured = result?.structuredContent ? JSON.stringify(result.structuredContent) : "";
  const context = text || structured || (typeof result === "string" ? result : JSON.stringify(result));
  if (!context) throw new Error("DuckDuckGo MCP returned no search content.");

  const sources = [];
  const markdownLinks = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;
  let match;
  while ((match = markdownLinks.exec(context)) !== null && sources.length < 6) sources.push({ title: match[1], url: match[2] });
  const urls = [...context.matchAll(/https?:\/\/[^\s)>]+/g)].map((item) => item[0].replace(/[.,;]+$/, ""));
  for (const url of urls) {
    if (sources.some((source) => source.url === url)) continue;
    sources.push({ title: url, url });
    if (sources.length >= 6) break;
  }
  return { sources, context };
}
async function executeRequestedTool(name, args) {
  if (name === "mcp_tool") {
    const serverName = String(args?.server || "");
    const toolName = String(args?.tool || "");
    const entry = state.mcpTools.find((item) => item.server.name === serverName && item.tool.name === toolName);
    if (!entry) throw new Error(`MCP tool not found: ${serverName}/${toolName}`);
    return (await mcpRequest(entry.server, "tools/call", { name: toolName, arguments: args.arguments || {} }, entry.sessionId)).result;
  }
  return executeMcpTool(name, args);
}

async function askModel(model, messages, tools) {
  const conversation = messages.map((message) => ({ ...message }));
  const maxToolRounds = 4;

  for (let round = 0; round <= maxToolRounds; round++) {
    const response = await fetch(apiUrl("/v1/chat/completions"), {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ model, messages: conversation, tools: tools.length ? tools : undefined }),
    });
    const raw = await response.text();
    let payload = {};
    try { payload = JSON.parse(raw); } catch { /* preserve the status below */ }
    if (!response.ok) throw new Error(payload.error?.message || raw.trim().slice(0, 500) || `Request failed (${response.status})`);
    const choice = payload.choices?.[0];
    const toolCalls = choice?.message?.tool_calls;

    if (Array.isArray(toolCalls) && toolCalls.length) {
      if (round === maxToolRounds) throw new Error("Model exceeded the tool-call limit.");
      conversation.push({ role: "assistant", content: choice.message.content || null, tool_calls: toolCalls });
      for (const call of toolCalls) {
        let args = {};
        try { args = JSON.parse(call.function?.arguments || "{}"); } catch { throw new Error(`Invalid arguments for tool ${call.function?.name || "unknown"}.`); }
        const result = await executeRequestedTool(call.function?.name || "", args);
        conversation.push({ role: "tool", tool_call_id: call.id, name: call.function?.name, content: JSON.stringify(result) });
      }
      continue;
    }

    const content = choice?.message?.content;
    if (typeof content !== "string" || !content.trim()) throw new Error(`Empty model response${choice?.finish_reason ? ` (${choice.finish_reason})` : ""}.`);
    return content;
  }

  throw new Error("Model did not return a response.");
}
function requestedTools() {
  const tools = [];
  if (state.mcpTools.length && $("#mcpToggle").getAttribute("aria-pressed") === "true") {
    tools.push({ type: "function", function: { name: "mcp_tool", description: "Request an approved tool from a configured MCP server. The host application must execute this tool.", parameters: { type: "object", properties: { server: { type: "string" }, tool: { type: "string" }, arguments: { type: "object" } }, required: ["server", "tool", "arguments"] } } });
    for (const entry of state.mcpTools) tools.push({ type: "function", function: { name: entry.tool.name, description: `[${entry.server.name}] ${entry.tool.description || "MCP tool"}`, parameters: entry.tool.inputSchema || { type: "object", properties: {} } } });
  }
  return tools;
}

async function autoSendMessageLegacy() {
  if (!prompt) return;
  if (!state.relayUrl || !state.relayKey) return openSettings();
  const models = state.selected.length ? state.selected : state.models.slice(0, 1).map((model) => model.id);
  if (!models.length) return openSettings();

  els.prompt.value = "";
  state.messages.push({ kind: "user", content: prompt });
  renderConversation();

  const searchEnabled = $("#searchToggle").getAttribute("aria-pressed") === "true";
  let search = { sources: [], context: "" };
  if (searchEnabled) {
    const searchMessage = { kind: "search", content: "Searching the web...", label: prompt };
    state.messages.push(searchMessage);
    renderConversation();
    try {
      search = await duckDuckGoMcpSearch(prompt);
      searchMessage.content = `Found ${search.sources.length} source${search.sources.length === 1 ? "" : "s"}.`;
      searchMessage.sources = search.sources;
      search.context = search.context.slice(0, 12000);
    } catch (error) {
      searchMessage.content = error instanceof Error ? error.message : String(error);
      searchMessage.error = true;
    }
    renderConversation();
  }

  const tools = requestedTools();
  const history = state.messages.filter((message) => message.kind === "user" || message.kind === "assistant").map((message) => ({ role: message.kind, content: message.content }));
  const research = search.context ? { role: "system", content: `Use these web search results as current evidence. Cite sources as [N] where relevant and do not invent details.\n\n${search.context}` } : null;
  const modelMessages = research ? [research, ...history] : history;
  state.messages.push({ kind: "compare", items: models.map((model) => ({ model, content: "Waiting..." })) });
  renderConversation();
  const compareMessage = state.messages[state.messages.length - 1];
  const results = await Promise.all(models.map(async (model) => {
    try { return { model, content: await askModel(model, modelMessages, tools) }; }
    catch (error) { return { model, content: error instanceof Error ? error.message : String(error), error: true }; }
  }));
  compareMessage.items = results;
  renderConversation();
  const usable = results.filter((result) => !result.error);
  if (!usable.length) return;
  const synthesisModel = els.synthesisModel.value || usable[0].model;
  const evidence = usable.map((result) => `MODEL: ${result.model}\n${result.content.slice(0, 6000)}`).join("\n\n---\n\n").slice(0, 18000);
  try {
    const synthesis = await askModel(synthesisModel, [
      { role: "user", content: `Synthesize the candidate answers below. Preserve useful disagreement, answer the original request directly, and use the supplied web sources when present.\n\nOriginal request:\n${prompt}\n\n${search.context ? `Web sources:\n${search.context}\n\n` : ""}Candidate answers:\n${evidence}` },
    ], []);
    state.messages.push({ kind: "assistant", content: synthesis, label: synthesisModel, sources: search.sources });
  } catch (error) {
    state.messages.push({ kind: "assistant", content: error instanceof Error ? error.message : String(error), error: true });
  }
  renderConversation();
}

async function prepareRequest() {
  const prompt = els.prompt.value.trim();
  if (!prompt) throw new Error("Enter a request first.");
  if (!state.relayUrl || !state.relayKey) { openSettings(); throw new Error("Relay settings are required."); }
  els.prompt.value = "";
  state.messages.push({ kind: "user", content: prompt });
  const priorContext = state.replyContext === "synthesized" ? state.lastSynthesis : state.lastComparison?.filter((item) => !item.error).map((item) => `MODEL: ${item.model}\n${item.content}`).join("\n\n---\n\n") || "";
  state.request = { prompt, search: { sources: [], context: "" }, priorContext, webSearch: $("#searchToggle").getAttribute("aria-pressed") === "true" };
  state.lastComparison = null;
  renderConversation();
  return state.request;
}

function modelMessages(request, search = request.search) {
  const history = [{ role: "user", content: request.prompt }];
  const context = request.priorContext ? `\n\nPrevious context from ${state.replyContext === "synthesized" ? "the selected synthesized viewpoint" : "the selected model viewpoints"}:\n${request.priorContext.slice(0, 12000)}` : "";
  return search.context ? [{ role: "system", content: `Use these web search results and prior context as evidence. Cite sources as [N] where relevant and do not invent details.\n\n${search.context.slice(0, 12000)}${context}` }, ...history] : context ? [{ role: "system", content: `Use this prior context to make the next answer more accurate.\n${context}` }, ...history] : history;
}

async function answerModelIndependently(model, request) {
  if (!request.webSearch) {
    return { model, content: await askModel(model, modelMessages(request), requestedTools()), sources: [] };
  }
  const query = (await askModel(model, [{ role: "user", content: `Create one concise DuckDuckGo search query that would help answer the request below. Return only the query, without quotes or explanation.\n\nRequest: ${request.prompt}` }], [])).trim().replace(/^['"]|['"]$/g, "").slice(0, 300);
  const search = await duckDuckGoMcpSearch(query);
  search.context = search.context.slice(0, 12000);
  const content = await askModel(model, modelMessages(request, search), requestedTools());
  return { model, content, searchQuery: query, sources: search.sources };
}

async function sendMessage() {
  return compareResponses();
}

async function compareResponses() {
  try {
    const request = state.request || await prepareRequest();
    const models = state.selected.length ? state.selected : state.models.slice(0, 3).map((model) => model.id);
    if (!models.length) throw new Error("Select at least one model.");
    const compareMessage = { kind: "compare", items: models.map((model) => ({ model, content: "Waiting..." })) };
    state.messages.push(compareMessage);
    renderConversation();
    compareMessage.items = await Promise.all(models.map(async (model) => {
      try { return await answerModelIndependently(model, request); }
      catch (error) { return { model, content: error instanceof Error ? error.message : String(error), error: true, sources: [] }; }
    }));
    state.lastComparison = compareMessage.items;
    request.search = {
      context: compareMessage.items.filter((item) => !item.error && item.searchQuery).map((item) => `${item.model} searched: ${item.searchQuery}`).join("\n"),
      sources: compareMessage.items.flatMap((item) => item.sources || []).filter((source, index, all) => all.findIndex((candidate) => candidate.url === source.url) === index),
    };
    renderConversation();
  } catch (error) {
    state.messages.push({ kind: "assistant", content: error instanceof Error ? error.message : String(error), error: true });
    renderConversation();
  }
}

async function synthesizeResponse() {
  try {
    const request = state.request;
    if (!request) throw new Error("Ask or compare a request before synthesizing.");
    const evidence = state.lastComparison?.length
      ? state.lastComparison.filter((item) => !item.error).map((item) => `MODEL: ${item.model}\n${item.content.slice(0, 6000)}`).join("\n\n---\n\n")
      : state.messages.filter((message) => message.kind === "assistant" && !message.error).slice(-6).map((message) => message.content.slice(0, 6000)).join("\n\n---\n\n");
    if (!evidence) throw new Error("There are no usable model responses to synthesize.");
    const model = state.synthesisModel || state.selected[0] || state.models[0]?.id;
    if (!model) throw new Error("Select a synthesis model.");
    const content = await askModel(model, [{ role: "user", content: `Synthesize these candidate answers. Answer the original request directly and preserve useful disagreement.\n\nOriginal request:\n${request.prompt}\n\n${request.search.context ? `Web sources:\n${request.search.context}\n\n` : ""}Candidate answers:\n${evidence.slice(0, 18000)}` }], []);
    state.lastSynthesis = content;
    state.messages.push({ kind: "assistant", content, label: `Combine · ${model}`, sources: request.search.sources });
    state.activeViewpointIndex = state.messages.length - 1;
    renderConversation();
    selectViewpoint(state.activeViewpointIndex);
  } catch (error) {
    state.messages.push({ kind: "assistant", content: error instanceof Error ? error.message : String(error), error: true });
    renderConversation();
  }
}
async function compareViewpoints() {
  try {
    const request = state.request;
    if (!request) throw new Error("Ask a request before comparing viewpoints.");
    const evidence = state.lastComparison?.filter((item) => !item.error).map((item) => `MODEL: ${item.model}\n${item.content.slice(0, 6000)}`).join("\n\n---\n\n");
    if (!evidence) throw new Error("There are no model viewpoints to compare.");
    const model = state.synthesisModel || state.selected[0] || state.models[0]?.id;
    if (!model) throw new Error("Select a comparison model.");
    const content = await askModel(model, [{ role: "user", content: `Compare the viewpoints below for the original request. Identify agreements, disagreements, tradeoffs, and which conclusions are best supported. Do not merely repeat the answers.\n\nOriginal request:\n${request.prompt}\n\nViewpoints:\n${evidence.slice(0, 18000)}` }], []);
    state.messages.push({ kind: "assistant", content, label: `Comparison · ${model}` });
    state.activeViewpointIndex = state.messages.length - 1;
    renderConversation();
    selectViewpoint(state.activeViewpointIndex);
  } catch (error) {
    state.messages.push({ kind: "assistant", content: error instanceof Error ? error.message : String(error), error: true });
    renderConversation();
  }
}

function busyAction(selector, busyLabel, action) {
  const button = $(selector);
  return async () => {
    if (button.disabled) return;
    const original = button.textContent;
    button.disabled = true;
    button.textContent = busyLabel;
    try { await action(); }
    finally { button.disabled = false; button.textContent = original; }
  };
}

$("#themeToggle").addEventListener("click", () => { state.theme = state.theme === "dark" ? "light" : "dark"; localStorage.setItem("theme", state.theme); applyTheme(); });
$("#resultList").addEventListener("click", (event) => {
  const button = event.target.closest("[data-result-index]");
  if (button) selectViewpoint(Number(button.dataset.resultIndex));
});
$("#historyList").addEventListener("click", (event) => {
  const deleteButton = event.target.closest("[data-delete-conversation]");
  if (deleteButton) {
    deleteConversation(deleteButton.dataset.deleteConversation);
    return;
  }
  const button = event.target.closest("[data-conversation-id]");
  if (button) loadConversation(button.dataset.conversationId);
});
$("#replyViewpoint").addEventListener("click", () => {
  const result = state.messages[state.activeViewpointIndex];
  if (!result) return;
  state.lastSynthesis = result.content;
  state.replyContext = "synthesized";
  els.prompt.placeholder = `Reply using ${result.label} as context...`;
  els.prompt.focus();
});
$("#closeViewpoint").addEventListener("click", () => setViewpointVisibility(false));
$("#toggleViewpoint").addEventListener("click", () => {
  const visible = els.viewpointPanel.classList.contains("open");
  if (visible) {
    setViewpointVisibility(false);
    return;
  }
  if (state.activeViewpointIndex !== null) {
    selectViewpoint(state.activeViewpointIndex);
    return;
  }
  setViewpointVisibility(true);
});
applyTheme();
$("#settingsForm").addEventListener("submit", saveSettings);
$("#openSettings").addEventListener("click", openSettings);
$("#topSettings").addEventListener("click", openSettings);
$("#openSidebar").addEventListener("click", () => els.sidebar.classList.toggle("open"));
$("#refreshModels").addEventListener("click", loadModels);
$("#sendButton").addEventListener("click", busyAction("#sendButton", "Asking...", sendMessage));
$("#compareButton").addEventListener("click", busyAction("#compareButton", "Comparing...", compareViewpoints));
$("#combineButton").addEventListener("click", busyAction("#combineButton", "Combining...", synthesizeResponse));
$("#prompt").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); } });
$("#searchToggle").addEventListener("click", (event) => toggle(event.currentTarget));
$("#mcpToggle").addEventListener("click", (event) => toggle(event.currentTarget));
$("#newChat").addEventListener("click", () => {
  createConversation();
  els.viewpointTitle.textContent = "No viewpoint selected";
  els.viewpointContent.textContent = "Use Combine or Compare to create a synthesized viewpoint.";
  els.replyViewpoint.disabled = true;
  els.sidebar.classList.remove("open");
});
document.querySelectorAll(".starter").forEach((button) => button.addEventListener("click", () => { els.prompt.value = button.dataset.prompt; els.prompt.focus(); }));
renderConversation();
loadModels();
discoverMcpTools();

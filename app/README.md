# Relay Studio client

This folder contains the mobile- and desktop-friendly client for the relay worker.

## Run locally

From the repository root:

```sh
python3 -m http.server 4173 --directory app
```

Open <http://localhost:4173> and configure:

- Relay base URL, with or without the `/v1` suffix
- Relay API key or Clerk user ID
- Optional synthesis model

Settings are saved in the app's local storage so they survive a desktop or Android app restart. Credentials should still be treated as secrets; use gateway mode for shared deployments.

The app discovers models from `GET /v1/models`, sends selected models in parallel through `POST /v1/chat/completions`, and sends the successful responses to a final synthesis model.

## MCP integration

DuckDuckGo search is embedded in the client as a built-in MCP-compatible adapter and requires no server configuration. Additional remote Streamable HTTP MCP servers are optional; add them in Settings as JSON:

```json
[
  {
    "name": "my-tools",
    "url": "https://example.com/mcp",
    "headers": {
      "Authorization": "Bearer <short-lived-token>"
    }
  }
]
```

The client initializes each optional server, lists its tools, and includes the discovered schemas in relay requests when MCP tools are enabled. Built-in DuckDuckGo search executes locally through the embedded adapter. Tool execution for additional servers should be restricted to trusted endpoints; the relay shim can return a tool call, but the client still needs a complete approval-and-follow-up loop before side-effecting tools are production-ready.

## Native builds

The repository currently contains the shared client source, but this environment does not have Flutter, Rust, Java, or the Android SDK installed. Therefore no verified `.AppImage` or `.apk` artifact has been generated yet. A native wrapper should package `app/` with Tauri 2 or Flutter, then build on a runner with Android and Linux toolchains.

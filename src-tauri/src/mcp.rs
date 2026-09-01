use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::Manager;

fn runtime_path(app: &tauri::AppHandle, name: &str) -> Result<PathBuf, String> {
    let path = app
        .path()
        .resource_dir()
        .map_err(|error| format!("Unable to locate bundled runtimes: {error}"))?
        .join("runtimes")
        .join(name);
    if path.is_file() {
        return Ok(path);
    }
    // Development fallback: use uvx from PATH. Release packages should bundle
    // a target-compatible uvx under resources/runtimes/.
    if name == "uvx" {
        return Ok(PathBuf::from(name));
    }
    Err(format!("Bundled runtime is missing: {}", path.display()))
}

fn send_json(writer: &mut impl Write, value: &Value) -> Result<(), String> {
    writeln!(writer, "{}", serde_json::to_string(value).map_err(|error| error.to_string())?)
        .map_err(|error| format!("Unable to write MCP request: {error}"))?;
    writer
        .flush()
        .map_err(|error| format!("Unable to flush MCP request: {error}"))
}

fn read_response(reader: &mut impl BufRead, expected_id: u64) -> Result<Value, String> {
    for line in reader.lines() {
        let line = line.map_err(|error| format!("Unable to read MCP response: {error}"))?;
        let value: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if value.get("id").and_then(Value::as_u64) == Some(expected_id) {
            if let Some(error) = value.get("error") {
                return Err(format!("DuckDuckGo MCP error: {error}"));
            }
            return Ok(value.get("result").cloned().unwrap_or(Value::Null));
        }
    }
    Err("DuckDuckGo MCP exited before returning a response.".to_string())
}

fn search_blocking(
    app: tauri::AppHandle,
    query: String,
    max_results: u32,
    region: String,
) -> Result<String, String> {
    if query.trim().is_empty() {
        return Err("Search query cannot be empty.".to_string());
    }
    let uvx = runtime_path(&app, "uvx")?;
    let mut child = Command::new(uvx)
        .args([
            "--with",
            "duckduckgo-mcp-server[browser]",
            "duckduckgo-mcp-server",
            "--search-backend",
            "auto",
        ])
        .env("DDG_SEARCH_BACKEND", "auto")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .env_remove("VIRTUAL_ENV")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Unable to start DuckDuckGo MCP: {error}"))?;

    let mut stdin = child.stdin.take().ok_or("DuckDuckGo MCP stdin unavailable.")?;
    let stdout = child.stdout.take().ok_or("DuckDuckGo MCP stdout unavailable.")?;
    let mut stderr = child.stderr.take().ok_or("DuckDuckGo MCP stderr unavailable.")?;
    let mut reader = BufReader::new(stdout);

    let read_or_report = |reader: &mut BufReader<_>, expected_id: u64, child: &mut std::process::Child, stderr: &mut std::process::ChildStderr| {
        match read_response(reader, expected_id) {
            Ok(value) => Ok(value),
            Err(error) => {
                let _ = child.kill();
                let mut detail = String::new();
                let _ = stderr.read_to_string(&mut detail);
                let detail = detail.trim();
                Err(if detail.is_empty() { error } else { format!("{error}: {detail}") })
            }
        }
    };

    send_json(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": { "name": "relay-studio", "version": "0.1.0" }
            }
        }),
    )?;
    read_or_report(&mut reader, 1, &mut child, &mut stderr)?;
    send_json(
        &mut stdin,
        &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
    )?;
    send_json(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": query,
                    "max_results": max_results,
                    "region": region
                }
            }
        }),
    )?;
    let result = read_or_report(&mut reader, 2, &mut child, &mut stderr)?;
    let text = result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n\n");

    let _ = child.kill();
    if text.trim().is_empty() {
        return Err("DuckDuckGo MCP returned no search results.".to_string());
    }
    Ok(text)
}

#[tauri::command]
pub async fn duckduckgo_search(
    app: tauri::AppHandle,
    query: String,
    max_results: Option<u32>,
    region: Option<String>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        search_blocking(app, query, max_results.unwrap_or(10).min(50), region.unwrap_or_default())
    })
    .await
    .map_err(|error| format!("DuckDuckGo MCP task failed: {error}"))?
}

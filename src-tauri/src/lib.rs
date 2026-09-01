#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod mcp;
mod runtime;

use mcp::duckduckgo_search;
use runtime::{runtime_status, RuntimeStatus};

#[tauri::command]
fn get_runtime_status(app: tauri::AppHandle) -> Result<RuntimeStatus, String> {
    runtime_status(&app)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![get_runtime_status, duckduckgo_search])
        .run(tauri::generate_context!())
        .expect("error while running Relay Studio");
}

use serde::Serialize;
use std::path::PathBuf;
use tauri::Manager;

#[derive(Debug, Serialize)]
pub struct RuntimeStatus {
    pub python: bool,
    pub uv: bool,
    pub uvx: bool,
    pub node: bool,
    pub runtime_dir: String,
}

fn runtime_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .resource_dir()
        .map(|path| path.join("runtimes"))
        .map_err(|error| format!("Unable to locate bundled runtimes: {error}"))
}

pub fn runtime_status(app: &tauri::AppHandle) -> Result<RuntimeStatus, String> {
    let directory = runtime_dir(app)?;
    let executable = |name: &str| {
        let path = directory.join(name);
        path.is_file()
    };

    Ok(RuntimeStatus {
        python: executable("python3"),
        uv: executable("uv"),
        uvx: executable("uvx"),
        node: executable("node"),
        runtime_dir: directory.display().to_string(),
    })
}

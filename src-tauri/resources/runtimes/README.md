# Bundled runtimes

The native package expects these executable files in this directory:

- `uv`
- `uvx`
- `python3`
- `node`

The bundled `uvx` requires the matching sibling `uv` executable. The DuckDuckGo MCP package is fetched into uv's cache on first use; release packaging therefore requires network access unless the uv cache is pre-populated.

The files must be built for the target platform and architecture. Do not commit binaries or credentials to git.

Linux AppImage targets should place executable files directly in this directory. Android requires ABI-specific native packaging and a bridge before these runtimes can be launched as local MCP servers; the Tauri project currently exposes runtime detection but does not claim that arbitrary Android subprocess execution is available.

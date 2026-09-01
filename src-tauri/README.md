# Native build

This directory contains the Tauri 2 native wrapper for the shared client in `app/`.

## Install prerequisites

Linux AppImage builds require Rust, Cargo, the Tauri Linux system dependencies, and an AppImage toolchain. Android builds require Rust, Java, Android Studio command-line tools, the Android SDK/NDK, and the target ABI configuration.

The current workspace does not have these prerequisites installed, so `.AppImage` and `.apk` artifacts cannot be generated here.

## Development

From the repository root:

```sh
npm install
npx tauri dev --config src-tauri/tauri.conf.json
```

## Linux AppImage

Place target-built `python3`, `uvx`, and `node` binaries in `src-tauri/resources/runtimes/`, then run:

```sh
npx tauri build --config src-tauri/tauri.conf.json --bundles appimage
```

The resulting AppImage is written under `src-tauri/target/release/bundle/appimage/`.

Run it directly when FUSE is installed:

```sh
chmod +x "src-tauri/target/release/bundle/appimage/Relay Studio_0.1.0_amd64.AppImage"
"src-tauri/target/release/bundle/appimage/Relay Studio_0.1.0_amd64.AppImage"
```

If the system reports `No suitable fusermount binary found`, install FUSE or use the included fallback launcher, which runs the AppImage in extracted mode:

```sh
./scripts/run-relay-studio.sh
```

On Debian/Ubuntu, the FUSE package is usually:

```sh
sudo apt install -y fuse3
```

## Android APK

Initialize Android support once:

```sh
npx tauri android init
npx tauri android build --apk
```

Android runtime support must be validated per ABI. The Linux executable layout cannot be copied directly into an APK. Python, `uvx`, and Node need Android-compatible native builds and a native process/runtime bridge. Remote Streamable HTTP MCP remains the fallback for servers which cannot run locally on Android.

## MCP runtime contract

The native layer starts the exact `duckduckgo-mcp-server` package with `uvx --with duckduckgo-mcp-server[browser]`, initializes it over stdio JSON-RPC, and calls its `search` tool through the `duckduckgo_search` Tauri command. Release packages should include a target-compatible `uvx` executable at `resources/runtimes/uvx`; development builds fall back to `uvx` on `PATH`. The process is short-lived per search and its stderr is discarded; no user-provided shell arguments are evaluated.

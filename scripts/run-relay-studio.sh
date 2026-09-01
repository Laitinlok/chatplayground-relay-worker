#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APPIMAGE=${1:-"$SCRIPT_DIR/../src-tauri/target/release/bundle/appimage/Relay Studio_0.1.0_amd64.AppImage"}
shift 2>/dev/null || true

if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -z "${DISPLAY:-}" ]; then
  export GDK_BACKEND="${GDK_BACKEND:-wayland}"
fi

if [ "${RELAY_STUDIO_USE_FUSE:-0}" = "1" ]; then
  exec "$APPIMAGE" "$@"
fi

EXTRACT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/relay-studio.XXXXXX")
cleanup() { rm -rf "$EXTRACT_DIR"; }
trap cleanup EXIT INT TERM
(
  cd "$EXTRACT_DIR"
  "$APPIMAGE" --appimage-extract >/dev/null
)
exec "$EXTRACT_DIR/squashfs-root/AppRun" "$@"

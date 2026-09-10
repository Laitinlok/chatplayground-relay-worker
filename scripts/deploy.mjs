import { spawnSync } from "node:child_process";

const CHAT_CACHE_BINDING = "CHAT_CACHE";

function runWrangler(args, options = {}) {
  const result = spawnSync("npx", ["wrangler", ...args], {
    encoding: "utf8",
    stdio: options.capture ? ["ignore", "pipe", "inherit"] : "inherit",
  });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
  return result.stdout ?? "";
}

function parseKeyList(output) {
  const parsed = JSON.parse(output);
  return Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed.result)
      ? parsed.result
      : [];
}

const keys = parseKeyList(
  runWrangler(
    ["kv", "key", "list", "--binding", CHAT_CACHE_BINDING],
    { capture: true },
  ),
);
for (const entry of keys) {
  if (typeof entry?.name !== "string") continue;
  runWrangler(["kv", "key", "delete", "--binding", CHAT_CACHE_BINDING, entry.name]);
}

runWrangler(["deploy", "--minify"]);

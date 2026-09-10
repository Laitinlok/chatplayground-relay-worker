import { Hono } from "hono";
import { createMiddleware } from "hono/factory";
import {
  type ApiKeyRow,
  type CreatedKey,
  createKey,
  deleteKey,
  listKeys,
  revokeKey,
} from "../services/api-keys";
import type { Env, Variables } from "../types/env";
import { unauthorized } from "../utils/errors";

/**
 * Admin key portal at /admin/keys on the relay hostname.
 *
 * Edge protection: Cloudflare Access self-hosted app on path /admin/*
 * (relay /v1/* stays public). The Worker verifies the signed Access identity
 * from the execution context against ADMIN_ALLOWED_EMAIL.
 */
const adminAccess = createMiddleware<{
  Bindings: Env;
  Variables: Variables;
}>(async (c, next) => {
  const allowed = c.env.ADMIN_ALLOWED_EMAIL?.trim().toLowerCase();
  if (!allowed) {
    // No email filter configured — skip the in-worker email check.
    // Access is still gating the path at the edge.
    await next();
    return;
  }
  const identity = await (
    c.executionCtx as ExecutionContext
  ).access?.getIdentity();
  const email = identity?.email?.trim().toLowerCase();
  if (!email || email !== allowed) {
    throw unauthorized("Cloudflare Access identity required for admin portal");
  }
  c.set("adminEmail", email);
  await next();
});

const admin = new Hono<{ Bindings: Env; Variables: Variables }>();

admin.use("*", adminAccess);

admin.get("/keys", async (c) => {
  if (!c.env.DB) {
    return c.text("D1 binding DB is not configured", 503);
  }
  const keys = await listKeys(c.env.DB);
  const email = c.get("adminEmail") ?? "";
  return c.html(renderPortal(keys, email));
});

admin.get("/keys.json", async (c) => {
  if (!c.env.DB) {
    return c.json({ error: "D1 binding DB is not configured" }, 503);
  }
  const keys = await listKeys(c.env.DB);
  return c.json({
    keys: keys.map((k) => ({
      id: k.id,
      key_prefix: k.key_prefix,
      label: k.label,
      created_at: k.created_at,
      revoked_at: k.revoked_at,
    })),
  });
});

admin.post("/keys", async (c) => {
  if (!c.env.DB) {
    return c.json({ error: "D1 binding DB is not configured" }, 503);
  }

  const contentType = c.req.header("content-type") ?? "";
  let label = "default";
  if (contentType.includes("application/json")) {
    const body = (await c.req.json().catch(() => ({}))) as { label?: string };
    if (typeof body.label === "string" && body.label.trim()) {
      label = body.label.trim().slice(0, 128);
    }
  } else {
    const form = await c.req.parseBody();
    const raw = form.label;
    if (typeof raw === "string" && raw.trim()) {
      label = raw.trim().slice(0, 128);
    }
  }

  const created = await createKey(c.env.DB, label);

  if (contentType.includes("application/json")) {
    return c.json(
      {
        id: created.id,
        key: created.key,
        key_prefix: created.key_prefix,
        label: created.label,
        created_at: created.created_at,
        warning: "Store this key now. It will not be shown again.",
      },
      201,
    );
  }

  // HTML form POST: render the portal directly with the new key shown
  // server-side. Avoid a redirect so that Cloudflare Access cannot strip
  // the raw key from the query string before the page loads.
  const email = c.get("adminEmail") ?? "";
  const keys = await listKeys(c.env.DB);
  return c.html(renderPortal(keys, email, created), 201);
});

admin.post("/keys/:id/revoke", async (c) => {
  if (!c.env.DB) {
    return c.json({ error: "D1 binding DB is not configured" }, 503);
  }
  const id = c.req.param("id");
  const ok = await revokeKey(c.env.DB, id);
  const contentType = c.req.header("content-type") ?? "";
  const accept = c.req.header("accept") ?? "";
  if (
    contentType.includes("application/json") ||
    accept.includes("application/json")
  ) {
    return ok
      ? c.json({ revoked: true, id })
      : c.json({ error: "Key not found" }, 404);
  }
  return c.redirect("/admin/keys", 303);
});

admin.post("/keys/:id/delete", async (c) => {
  if (!c.env.DB) {
    return c.json({ error: "D1 binding DB is not configured" }, 503);
  }
  const id = c.req.param("id");
  const ok = await deleteKey(c.env.DB, id);
  const contentType = c.req.header("content-type") ?? "";
  const accept = c.req.header("accept") ?? "";
  if (
    contentType.includes("application/json") ||
    accept.includes("application/json")
  ) {
    return c.json({ deleted: ok, id });
  }
  return c.redirect("/admin/keys", 303);
});

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderPortal(
  keys: ApiKeyRow[],
  email: string,
  newKey?: CreatedKey,
): string {
  const rows = keys
    .map((k) => {
      const status = k.revoked_at ? "revoked" : "active";
      const created = new Date(k.created_at).toISOString();
      const revokeBtn = k.revoked_at
        ? ""
        : `<form method="post" action="/admin/keys/${escapeHtml(k.id)}/revoke" style="display:inline" onsubmit="return confirm('Revoke this key?');">
            <button type="submit">Revoke</button>
          </form>`;
      const deleteBtn = `<form method="post" action="/admin/keys/${escapeHtml(k.id)}/delete" style="display:inline" onsubmit="return confirm('Permanently remove this key? This cannot be undone.');">
            <button class="remove" type="submit">Remove</button>
          </form>`;
      return `<tr>
        <td><code>${escapeHtml(k.key_prefix)}…</code></td>
        <td>${escapeHtml(k.label)}</td>
        <td>${escapeHtml(status)}</td>
        <td>${escapeHtml(created)}</td>
        <td>${revokeBtn}${deleteBtn}</td>
      </tr>`;
    })
    .join("\n");

  // Server-rendered banner — shown only on the response that contains the
  // newly created key. No query-string round-trip needed.
  const banner = newKey
    ? `<div class="banner">
    <strong>⚠ Copy this key now — it will not be shown again.</strong>
    <div style="margin-top:0.4rem">Label: <strong>${escapeHtml(newKey.label)}</strong></div>
    <div style="margin-top:0.25rem">ID: <code>${escapeHtml(newKey.id)}</code></div>
    <div style="margin-top:0.25rem">Key: <code id="new-key-value">${escapeHtml(newKey.key)}</code>
      <button type="button" onclick="navigator.clipboard.writeText(document.getElementById('new-key-value').textContent||'')" style="margin-left:0.5rem;font-size:0.8rem;padding:0.2rem 0.5rem;">Copy</button>
    </div>
  </div>`
    : "";

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Relay API keys</title>
  <style>
    :root { font-family: system-ui, sans-serif; color: #0f172a; background: #f8fafc; }
    body { max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.35rem; }
    .meta { color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem; }
    table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 2px rgb(0 0 0 / 6%); }
    th, td { text-align: left; padding: 0.65rem 0.75rem; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem; }
    th { background: #f1f5f9; font-weight: 600; }
    code { font-size: 0.85rem; }
    form.create { display: flex; gap: 0.5rem; margin: 1.25rem 0; flex-wrap: wrap; }
    input[type=text] { flex: 1; min-width: 12rem; padding: 0.5rem 0.65rem; border: 1px solid #cbd5e1; border-radius: 6px; }
    button { padding: 0.5rem 0.85rem; border: 0; border-radius: 6px; background: #0f172a; color: #fff; cursor: pointer; font-size: 0.875rem; }
    button:hover { background: #1e293b; }
    button.remove { background: #b91c1c; margin-left: 0.35rem; }
    button.remove:hover { background: #991b1b; }
    .banner { background: #fef3c7; border: 1px solid #f59e0b; color: #78350f; padding: 0.85rem 1rem; border-radius: 8px; margin-bottom: 1rem; word-break: break-all; }
    .banner strong { display: block; margin-bottom: 0.35rem; }
  </style>
</head>
<body>
  <h1>Relay API keys</h1>
  <p class="meta">Signed in via Cloudflare Access as <strong>${escapeHtml(email)}</strong>.
    Keys authenticate with the <code>Authorization: Bearer rly_…</code> header.
  </p>

  ${banner}

  <form class="create" method="post" action="/admin/keys">
    <input type="text" name="label" placeholder="Label (e.g. alice-laptop)" maxlength="128" required />
    <button type="submit">Create key</button>
  </form>

  <table>
    <thead>
      <tr><th>Prefix</th><th>Label</th><th>Status</th><th>Created</th><th></th></tr>
    </thead>
    <tbody>
      ${rows || '<tr><td colspan="5">No keys yet.</td></tr>'}
    </tbody>
  </table>
</body>
</html>`;
}

export default admin;

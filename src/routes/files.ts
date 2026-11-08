import { Hono } from "hono";
import { UPLOAD_TIMEOUT } from "../constants/timeouts";
import type { Env, Variables } from "../types/env";
import { invalidRequest, upstreamError } from "../utils/errors";

const files = new Hono<{ Bindings: Env; Variables: Variables }>();
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set([
  "pdf",
  "doc",
  "docx",
  "txt",
  "xls",
  "xlsx",
  "csv",
  "html",
  "js",
  "css",
  "py",
]);

// OpenAI Files API proxy. Validate the supplied file before forwarding it.
files.post("/v1/files", async (c) => {
  const contentType = c.req.header("content-type");
  if (!contentType?.toLowerCase().startsWith("multipart/form-data")) {
    throw invalidRequest(
      "Request must be multipart/form-data with a 'file' field.",
    );
  }

  if (!c.req.raw.body) throw invalidRequest("Request body is empty.");
  const contentLength = Number(c.req.header("content-length"));
  if (Number.isFinite(contentLength) && contentLength > MAX_FILE_BYTES + 64 * 1024) {
    throw invalidRequest("Files must be 10 MB or smaller.");
  }

  let form: FormData;
  try {
    form = await c.req.formData();
  } catch {
    throw invalidRequest("Request body must contain a valid multipart file.");
  }
  const uploaded = form.get("file");
  if (!(uploaded instanceof File)) {
    throw invalidRequest("Multipart form must include a 'file' field.");
  }
  if (uploaded.size === 0) throw invalidRequest("Uploaded file is empty.");
  const extension = uploaded.name.split(".").at(-1)?.toLowerCase() ?? "";
  if (!ALLOWED_EXTENSIONS.has(extension)) {
    throw invalidRequest(
      "Unsupported file type. Allowed: PDF, DOC/DOCX, TXT, XLS/XLSX, CSV, HTML, JS, CSS, PY.",
    );
  }
  if (uploaded.size > MAX_FILE_BYTES) {
    throw invalidRequest("Files must be 10 MB or smaller.");
  }
  const bytes = uploaded.size;
  const forward = new FormData();
  for (const [key, value] of form.entries()) forward.append(key, value);

  const res = await fetch(c.env.UPSTREAM_UPLOAD_URL, {
    method: "POST",
    headers: {
      origin: c.env.UPSTREAM_ORIGIN,
      referer: c.env.UPSTREAM_REFERER,
      accept: "*/*",
    },
    body: forward,
    signal: AbortSignal.timeout(UPLOAD_TIMEOUT),
  });

  if (!res.ok) {
    throw upstreamError(
      res.status,
      `Upload failed: upstream returned ${res.status}.`,
    );
  }

  const body = (await res.json().catch(() => null)) as { url?: string } | null;
  if (!body?.url || !body.url.startsWith("https://")) {
    throw upstreamError(
      502,
      "Upload succeeded but upstream did not return a URL.",
    );
  }

  // The returned URL doubles as image_url.url for follow-up vision chats.
  return c.json({
    id: body.url, // doubles as image_url.url for follow-up vision chats
    object: "file",
    bytes: Number.isFinite(bytes) ? bytes : 0,
    created_at: Math.floor(Date.now() / 1000),
    filename: uploaded.name,
    purpose: "vision",
    status: "processed",
    status_details: null,
  });
});

export default files;

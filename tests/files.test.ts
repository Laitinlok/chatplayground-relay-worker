import { Hono } from "hono";
import { afterEach, describe, expect, it, vi } from "vitest";
import { errorHandler } from "../src/middleware/error-handler";
import files from "../src/routes/files";

// /v1/files needs no upstream credential. Uploads are validated then proxied.

const env = {
  UPSTREAM_UPLOAD_URL: "https://up.example.test/api/upload",
  UPSTREAM_ORIGIN: "https://web.example.test",
  UPSTREAM_REFERER: "https://web.example.test/",
};

function post(body?: BodyInit, contentType?: string) {
  // oxlint-disable-next-line typescript/no-explicit-any -- minimal test env stub
  const app = new Hono<any>();
  app.onError(errorHandler);
  app.route("/", files);
  const headers = contentType ? { "content-type": contentType } : undefined;
  return app.request("/v1/files", { method: "POST", headers, body }, env);
}

function upload(name = "document.pdf", text = "file contents") {
  const form = new FormData();
  form.append("file", new File([text], name));
  return form;
}

async function envelope(res: Response) {
  const body = (await res.json()) as { error: { message: string } };
  return body.error;
}

afterEach(() => vi.restoreAllMocks());

describe("POST /v1/files", () => {
  it("rejects a non-multipart content-type without uploading", async () => {
    vi.spyOn(globalThis, "fetch");
    const res = await post("{}", "application/json");
    expect(res.status).toBe(400);
    expect((await envelope(res)).message).toContain("multipart/form-data");
    expect(vi.mocked(fetch)).not.toHaveBeenCalled();
  });

  it("rejects a multipart request with no file", async () => {
    vi.spyOn(globalThis, "fetch");
    const res = await post(new FormData());
    expect(res.status).toBe(400);
    expect((await envelope(res)).message).toContain("'file' field");
    expect(vi.mocked(fetch)).not.toHaveBeenCalled();
  });

  it("forwards the body and returns the URL as the file id", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({ url: "https://files.example.test/abc.pdf" }),
    );

    const res = await post(upload("photo.pdf"));
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({
      id: "https://files.example.test/abc.pdf",
      object: "file",
      bytes: 13,
      filename: "photo.pdf",
      purpose: "vision",
      status: "processed",
    });

    const [url, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
    expect(url).toBe(env.UPSTREAM_UPLOAD_URL);
    const headers = init.headers as Record<string, string>;
    expect(headers["content-type"]).toBeUndefined();
    expect(headers.origin).toBe(env.UPSTREAM_ORIGIN);
    expect(headers.authorization).toBeUndefined();
  });

  it("reports the uploaded file size", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({ url: "https://files.example.test/abc.png" }),
    );
    const res = await post(upload("notes.txt", "abc"));
    expect(((await res.json()) as { bytes: number }).bytes).toBe(3);
  });

  it("maps an upstream failure through upstreamError", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("", { status: 500 }),
    );
    const res = await post(upload());
    expect(res.status).toBe(502);
    expect((await envelope(res)).message).toContain("upstream returned 500");
  });

  it.each([
    "pdf", "doc", "docx", "txt", "xls", "xlsx", "csv", "html", "js", "css", "py",
  ])("accepts .%s uploads", async (ext) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({ url: "https://files.example.test/abc" }),
    );
    const res = await post(upload(`file.${ext}`));
    expect(res.status).toBe(200);
  });

  it("rejects unsupported formats", async () => {
    vi.spyOn(globalThis, "fetch");
    const res = await post(upload("file.exe"));
    expect(res.status).toBe(400);
    expect(vi.mocked(fetch)).not.toHaveBeenCalled();
  });

  it("502s when upstream accepts the upload but returns no URL", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({ ok: true }),
    );
    const res = await post(upload());
    expect(res.status).toBe(502);
    expect((await envelope(res)).message).toContain("did not return a URL");
  });
});

import { Hono } from "hono";
import { cors } from "hono/cors";
import { auth } from "./middleware/auth";
import { errorHandler } from "./middleware/error-handler";
import admin from "./routes/admin-keys";
import chat from "./routes/chat";
import files from "./routes/files";
import images from "./routes/images";
import embeddings from "./routes/embeddings";
import models from "./routes/models";
import responses from "./routes/responses";
import type { Env, Variables } from "./types/env";

const app = new Hono<{ Bindings: Env; Variables: Variables }>();

app.onError(errorHandler);

app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: [
      "Content-Type",
      "Authorization",
      "OpenAI-Beta",
      "X-Conversation-Id",
    ],
  }),
);

app.get("/", (c) =>
  c.json({
    name: "chatplayground",
    description:
      "OpenAI-compatible relay for chatplayground.ai (BYOK, stateless).",
    endpoints: [
      "/v1/models",
      "/v1/models/capabilities",
      "/v1/chat/completions",
      "/v1/responses",
      "/v1/files",
      "/v1/images/generations",
      "/v1/embeddings",
    ],
  }),
);

// All /v1/* requires a valid Clerk session JWT as Bearer.
// Note: /models/capabilities is allowed via auth middleware as well for clients that omit /v1.
app.use("/models/capabilities", auth);
app.use("/v1/*", auth);
app.route("/", models);
app.route("/", chat);
app.route("/", responses);
app.route("/", files);
app.route("/", images);
app.route("/", embeddings);
app.route("/admin", admin);

export default app;

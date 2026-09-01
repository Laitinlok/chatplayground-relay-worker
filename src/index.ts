import { Hono } from "hono";
import { cors } from "hono/cors";
import { auth } from "./middleware/auth";
import { errorHandler } from "./middleware/error-handler";
import chat from "./routes/chat";
import files from "./routes/files";
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
    endpoints: ["/v1/models", "/v1/chat/completions", "/v1/responses", "/v1/files"],
  }),
);

// All /v1/* requires a valid Clerk session JWT as Bearer.
app.use("/v1/*", auth);
app.route("/", models);
app.route("/", chat);
app.route("/", responses);
app.route("/", files);

export default app;

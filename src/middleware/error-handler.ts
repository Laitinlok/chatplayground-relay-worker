import type { ErrorHandler } from "hono";
import type { Env, Variables } from "../types/env";
import { OpenAIHTTPError } from "../utils/errors";

export const errorHandler: ErrorHandler<{
  Bindings: Env;
  Variables: Variables;
}> = (err, _c) => {
  if (err instanceof OpenAIHTTPError) {
    const headers = new Headers({ "content-type": "application/json" });
    if (err.status === 429) {
      // ChatPlayground does not provide Retry-After. Prevent compatible clients
      // from immediately replaying the same request and worsening the burst.
      headers.set("retry-after", "60");
      headers.set("x-should-retry", "false");
    }
    return new Response(JSON.stringify(err.toEnvelope()), {
      status: err.status,
      headers,
    });
  }

  console.error("Unhandled error", err);
  const message = err instanceof Error ? err.message : "Internal server error";
  return new Response(
    JSON.stringify({
      error: {
        message,
        type: "internal_error",
        code: null,
        param: null,
      },
    }),
    {
      status: 500,
      headers: { "content-type": "application/json" },
    },
  );
};

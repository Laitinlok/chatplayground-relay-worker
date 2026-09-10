import { describe, expect, it, vi } from "vitest";
import { deleteKey } from "../src/services/api-keys";

function dbWithChanges(changes: number): D1Database {
  const run = vi.fn().mockResolvedValue({ meta: { changes } });
  const bind = vi.fn().mockReturnValue({ run });
  const prepare = vi.fn().mockReturnValue({ bind });
  return { prepare } as unknown as D1Database;
}

describe("deleteKey", () => {
  it("permanently deletes a matching key", async () => {
    const db = dbWithChanges(1);

    await expect(deleteKey(db, "key-123")).resolves.toBe(true);
    expect(db.prepare).toHaveBeenCalledWith(
      "DELETE FROM api_keys WHERE id = ?1",
    );
  });

  it("reports a missing key without deleting another row", async () => {
    await expect(deleteKey(dbWithChanges(0), "missing")).resolves.toBe(false);
  });
});

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const webDir = process.cwd();
const repoRoot = path.resolve(webDir, "..");
const e2eDbPath =
  process.env.CHESS_ML_E2E_DB_PATH ?? path.join(os.tmpdir(), "chess_ml-playwright.sqlite3");
const missingMaiaDir = path.join(os.tmpdir(), "chess_ml-missing-maia");
const missingLc0Path = path.join(os.tmpdir(), "chess_ml-missing-lc0");

const baseEnv = Object.fromEntries(
  Object.entries(process.env).filter((entry): entry is [string, string] => {
    return typeof entry[1] === "string";
  }),
);

async function globalSetup(): Promise<void> {
  fs.rmSync(e2eDbPath, { force: true });

  const seedEnv = {
    ...baseEnv,
    CHESS_ML_DB_PATH: e2eDbPath,
    CHESS_ML_DEMO_STOCKFISH_DEPTH: "6",
    CHESS_ML_EXPLANATION_PROVIDER: "disabled",
    CHESS_ML_CLASSIFIER_CHECKPOINT: "",
    CHESS_ML_LC0_PATH: missingLc0Path,
    CHESS_ML_MAIA_WEIGHTS_DIR: missingMaiaDir,
    UV_CACHE_DIR: process.env.UV_CACHE_DIR ?? path.join(os.tmpdir(), "chess_ml_uv_cache"),
  } as unknown as NodeJS.ProcessEnv;

  const result = spawnSync("uv", ["run", "python", "-m", "chess_ml.profile.demo"], {
    cwd: repoRoot,
    env: seedEnv,
    stdio: "inherit",
  });

  if (result.status !== 0) {
    throw new Error("Failed to seed the Playwright demo database.");
  }
}

export default globalSetup;

import { defineConfig, devices } from "@playwright/test";
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

const e2eEnv: Record<string, string> = {
  ...baseEnv,
  CHESS_ML_DB_PATH: e2eDbPath,
  CHESS_ML_DEMO_STOCKFISH_DEPTH: "6",
  CHESS_ML_STOCKFISH_DEPTH: "4",
  CHESS_ML_STOCKFISH_WORKERS: "1",
  CHESS_ML_ANALYSIS_TIMEOUT_SECONDS: "20",
  CHESS_ML_EXPLANATION_PROVIDER: "disabled",
  CHESS_ML_CLASSIFIER_CHECKPOINT: "",
  CHESS_ML_LC0_PATH: missingLc0Path,
  CHESS_ML_MAIA_WEIGHTS_DIR: missingMaiaDir,
  CHESS_ML_PLAY_STOCKFISH_TIME_MS: "80",
  UV_CACHE_DIR: process.env.UV_CACHE_DIR ?? path.join(os.tmpdir(), "chess_ml_uv_cache"),
};

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  timeout: 90_000,
  expect: {
    timeout: 20_000,
  },
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: "uv run uvicorn chess_ml.api.main:app --port 8000",
      cwd: repoRoot,
      env: e2eEnv,
      url: "http://localhost:8000/health",
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: "npm run dev -- --port 3000",
      cwd: webDir,
      url: "http://localhost:3000",
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});

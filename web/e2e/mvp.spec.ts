import { expect, type Locator, type Page, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(process.cwd(), "..");
const samplePgn = fs.readFileSync(
  path.join(repoRoot, "tests", "fixtures", "demo", "missed-tactic.pgn"),
  "utf-8",
);
const gamesApiUrl = "http://localhost:8000/api/games";

test("uploads a sample PGN and shows deterministic review motifs", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("textbox", { name: "PGN" }).fill(samplePgn);
  await page.getByRole("button", { name: "Analyze game" }).click();

  await expect(page.getByRole("heading", { name: /Ada.*Turing/ })).toBeVisible();
  await expect(page.getByText("Missed tactic").first()).toBeVisible();
  await expect(page.getByText("Best", { exact: true })).toBeVisible();
  await expect(page.getByText("Line", { exact: true })).toBeVisible();
});

test("dashboard loads the seeded demo profile", async ({ page }) => {
  await page.goto("/dashboard");

  await expect(page.getByTestId("summary-games")).toContainText("3");
  await expect(page.getByTestId("summary-motifs")).toContainText(/Motifs[1-9][0-9]*/);
  await expect(page.getByText("Nina (1450) vs Omar (1500)")).toBeVisible();
  await expect(page.getByText("Ada (1500) vs Turing (1520)")).toBeVisible();
  await expect(page.getByText("Priya (1300) vs Max (1350)")).toBeVisible();
  await expect(page.getByText(/Missed tactic|Allowed tactic|Hanging piece/).first()).toBeVisible();
});

test("play starts with Stockfish fallback, resigns, and opens review", async ({ page }) => {
  await page.goto("/play");

  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByText("Stockfish fallback (1350 Elo)")).toBeVisible();
  await expect(page.getByText(/Maia was not available/)).toBeVisible();

  await movePiece(page, page.getByTestId("chess-board"), "e2", "e4");
  await expect(page.getByText("e4")).toBeVisible();

  await page.getByRole("button", { name: "Resign and review" }).click();
  await expect(page.getByRole("heading", { name: "Moves" })).toBeVisible();
  await expect(page.getByText("Best", { exact: true })).toBeVisible();
});

test("bad PGN shows clear user-facing copy", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("textbox", { name: "PGN" }).fill("not a pgn");
  await page.getByRole("button", { name: "Analyze game" }).click();

  await expect(page.locator('p[role="alert"]')).toContainText(
    "That PGN could not be read as a standard chess game.",
  );
});

test("network failure shows local API setup copy", async ({ page }) => {
  await page.route(gamesApiUrl, async (route) => {
    await route.abort("failed");
  });
  await page.goto("/");

  await page.getByRole("textbox", { name: "PGN" }).fill(samplePgn);
  await page.getByRole("button", { name: "Analyze game" }).click();

  await expect(page.locator('p[role="alert"]')).toContainText(
    "Cannot reach the local API at http://localhost:8000.",
  );
});

test("engine failures show Stockfish-specific copy", async ({ page }) => {
  await page.route(gamesApiUrl, async (route) => {
    await route.fulfill({
      status: 504,
      contentType: "application/json",
      body: JSON.stringify({
        error: {
          code: "analysis_timeout",
          message: "Stockfish analysis exceeded the wall-clock budget.",
          details: {},
        },
      }),
    });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "PGN" }).fill(samplePgn);
  await page.getByRole("button", { name: "Analyze game" }).click();
  await expect(page.locator('p[role="alert"]')).toContainText(
    "Stockfish took too long to analyze this game.",
  );

  await page.unroute(gamesApiUrl);
  await page.route(gamesApiUrl, async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({
        error: {
          code: "stockfish_unavailable",
          message: "Stockfish binary not found.",
          details: {},
        },
      }),
    });
  });
  await page.getByRole("textbox", { name: "PGN" }).fill(samplePgn);
  await page.getByRole("button", { name: "Analyze game" }).click();
  await expect(page.locator('p[role="alert"]')).toContainText("Stockfish is not available.");
});

async function movePiece(
  page: Page,
  board: Locator,
  fromSquare: string,
  toSquare: string,
): Promise<void> {
  const box = await board.boundingBox();
  if (!box) {
    throw new Error("Chessboard was not visible.");
  }
  const from = squareCenter(box, fromSquare);
  const to = squareCenter(box, toSquare);
  await page.mouse.click(from.x, from.y);
  await page.mouse.click(to.x, to.y);
}

function squareCenter(
  box: { x: number; y: number; width: number; height: number },
  square: string,
): { x: number; y: number } {
  const file = square.charCodeAt(0) - "a".charCodeAt(0);
  const rank = Number(square[1]);
  return {
    x: box.x + ((file + 0.5) * box.width) / 8,
    y: box.y + ((8 - rank + 0.5) * box.height) / 8,
  };
}

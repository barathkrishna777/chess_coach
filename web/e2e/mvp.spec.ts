import { expect, type Locator, type Page, type Route, test } from "@playwright/test";
import { Chess } from "chess.js";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(process.cwd(), "..");
const samplePgn = fs.readFileSync(
  path.join(repoRoot, "tests", "fixtures", "demo", "missed-tactic.pgn"),
  "utf-8",
);
const gamesApiUrl = "http://localhost:8000/api/games";
const playApiUrl = "http://localhost:8000/api/play";
const startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

test("uploads a sample PGN and shows deterministic review motifs", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("textbox", { name: "PGN" }).fill(samplePgn);
  await page.getByRole("button", { name: "Analyze game" }).click();

  await expect(page.getByRole("heading", { name: /Ada.*Turing/ })).toBeVisible();
  await expect(page.getByText("Missed tactic").first()).toBeVisible();
  await expect(page.getByText("Best", { exact: true })).toBeVisible();
  await expect(page.getByText("Main line", { exact: true })).toBeVisible();
});

test("arrow keys advance through moves on the review screen", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("textbox", { name: "PGN" }).fill(samplePgn);
  await page.getByRole("button", { name: "Analyze game" }).click();

  await expect(page.getByRole("heading", { name: "1. e4" })).toBeVisible();

  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("heading", { name: "1... d5" })).toBeVisible();

  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("heading", { name: "2. exd5" })).toBeVisible();

  await page.keyboard.press("ArrowLeft");
  await expect(page.getByRole("heading", { name: "1... d5" })).toBeVisible();
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

test("starts a personal drill from the dashboard and reveals feedback after a move", async ({
  page,
  request,
}) => {
  await page.goto("/dashboard");

  await page.getByRole("link", { name: "Drill this motif" }).first().click();
  await expect(page).toHaveURL(/\/train\?motif=/);
  const board = page.getByTestId("training-board");
  await expect(board).toBeVisible();
  await expect(page.getByTestId("revealed-best-move")).toHaveCount(0);

  const motif = new URL(page.url()).searchParams.get("motif") ?? "any";
  const drillResponse = await request.get(
    `http://localhost:8000/api/train/next?motif=${encodeURIComponent(motif)}`,
  );
  expect(drillResponse.ok()).toBeTruthy();
  const drill = (await drillResponse.json()) as { fen: string };
  const move = firstLegalMove(drill.fen);

  await movePiece(page, board, move.from, move.to, sideToMove(drill.fen));

  await expect(page.getByText(/Correct|Incorrect/)).toBeVisible();
  await expect(page.getByTestId("revealed-best-move")).toBeVisible();
});

test("play starts with Stockfish fallback, resigns, and opens review", async ({ page }) => {
  await page.goto("/play");

  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByText("Stockfish fallback (1350 Elo)")).toBeVisible();
  await expect(page.getByText(/Maia was not available/)).toBeVisible();

  await dragPiece(page, page.getByTestId("chess-board"), "e2", "e4");
  await expect(page.getByText("e4")).toBeVisible();

  await page.getByRole("button", { name: "Resign and review" }).click();
  await expect(page.getByRole("heading", { name: "Moves" })).toBeVisible();
  await expect(page.getByText("Best", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Play again" }).click();
  await expect(page.getByRole("button", { name: "New game" })).toBeVisible();
  await expect(page.getByText("In progress")).toBeVisible();
});

test("play accepts click movement without regressing board interaction", async ({ page }) => {
  let submittedUci: string | null = null;
  await mockPlayOpponents(page);
  await page.route(`${playApiUrl}/new`, async (route) => {
    await fulfillJson(route, playState());
  });
  await page.route(`${playApiUrl}/move`, async (route) => {
    const body = route.request().postDataJSON() as { uci: string };
    submittedUci = body.uci;
    await fulfillJson(
      route,
      playState({
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        moves: [
          { ply: 1, side: "white", san: "e4", uci: "e2e4" },
          { ply: 2, side: "black", san: "e5", uci: "e7e5" },
        ],
        bot_move: { uci: "e7e5", san: "e5" },
      }),
    );
  });

  await page.goto("/play");
  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByText("White to move")).toBeVisible();
  await movePiece(page, page.getByTestId("chess-board"), "e2", "e4");

  await expect.poll(() => submittedUci).toBe("e2e4");
  await expect(page.getByText("e4")).toBeVisible();
});

test("promotion uses an in-board dialog instead of a browser prompt", async ({ page }) => {
  let dialogShown = false;
  let submittedUci: string | null = null;
  page.on("dialog", async (dialog) => {
    dialogShown = true;
    await dialog.dismiss();
  });
  await mockPlayOpponents(page);
  await page.route(`${playApiUrl}/new`, async (route) => {
    await fulfillJson(
      route,
      playState({
        fen: "7k/P7/8/8/8/8/8/7K w - - 0 1",
        legal_moves: [
          {
            from_square: "a7",
            destinations: [{ to_square: "a8", promotions: ["q", "r", "b", "n"] }],
          },
        ],
      }),
    );
  });
  await page.route(`${playApiUrl}/move`, async (route) => {
    const body = route.request().postDataJSON() as { uci: string };
    submittedUci = body.uci;
    await fulfillJson(
      route,
      playState({
        fen: "Q6k/8/8/8/8/8/8/7K b - - 0 1",
        legal_moves: [],
        moves: [{ ply: 1, side: "white", san: "a8=Q+", uci: "a7a8q" }],
      }),
    );
  });

  await page.goto("/play");
  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByText("White to move")).toBeVisible();
  await movePiece(page, page.getByTestId("chess-board"), "a7", "a8");
  await expect(page.getByRole("dialog", { name: "Choose promotion piece" })).toBeVisible();
  await page.getByRole("button", { name: "Promote to Queen" }).click();

  await expect.poll(() => submittedUci).toBe("a7a8q");
  expect(dialogShown).toBe(false);
});

test("play as black flips the board and starts with a bot move", async ({ page }) => {
  let requestedColor: string | null = null;
  let submittedUci: string | null = null;
  await mockPlayOpponents(page);
  await page.route(`${playApiUrl}/new`, async (route) => {
    const body = route.request().postDataJSON() as { user_color?: string };
    requestedColor = body.user_color ?? null;
    await fulfillJson(
      route,
      playState({
        user_color: "black",
        orientation: "black",
        fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        legal_moves: [
          {
            from_square: "e7",
            destinations: [{ to_square: "e5", promotions: [] }],
          },
        ],
        moves: [{ ply: 1, side: "white", san: "e4", uci: "e2e4" }],
        bot_move: { uci: "e2e4", san: "e4" },
      }),
    );
  });
  await page.route(`${playApiUrl}/move`, async (route) => {
    const body = route.request().postDataJSON() as { uci: string };
    submittedUci = body.uci;
    await fulfillJson(
      route,
      playState({
        user_color: "black",
        orientation: "black",
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        legal_moves: [
          {
            from_square: "d7",
            destinations: [{ to_square: "d5", promotions: [] }],
          },
        ],
        moves: [
          { ply: 1, side: "white", san: "e4", uci: "e2e4" },
          { ply: 2, side: "black", san: "e5", uci: "e7e5" },
          { ply: 3, side: "white", san: "Nf3", uci: "g1f3" },
        ],
        bot_move: { uci: "g1f3", san: "Nf3" },
      }),
    );
  });

  await page.goto("/play");
  await page.getByLabel("Play as").selectOption("black");
  await page.getByRole("button", { name: "Start game" }).click();

  await expect.poll(() => requestedColor).toBe("black");
  await expect(page.getByText("Black to move")).toBeVisible();
  await expect(page.getByText("e4")).toBeVisible();

  await movePiece(page, page.getByTestId("chess-board"), "e7", "e5", "black");
  await expect.poll(() => submittedUci).toBe("e7e5");
});

test("takeback undoes the latest user and bot move once", async ({ page }) => {
  await mockPlayOpponents(page);
  await page.route(`${playApiUrl}/new`, async (route) => {
    await fulfillJson(route, playState());
  });
  await page.route(`${playApiUrl}/move`, async (route) => {
    await fulfillJson(
      route,
      playState({
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        moves: [
          { ply: 1, side: "white", san: "e4", uci: "e2e4" },
          { ply: 2, side: "black", san: "e5", uci: "e7e5" },
        ],
        bot_move: { uci: "e7e5", san: "e5" },
      }),
    );
  });
  await page.route(`${playApiUrl}/takeback`, async (route) => {
    await fulfillJson(route, playState({ takebacks_remaining: 0 }));
  });

  await page.goto("/play");
  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByText("White to move")).toBeVisible();
  await movePiece(page, page.getByTestId("chess-board"), "e2", "e4");
  await expect(page.getByText("e4")).toBeVisible();

  await page.getByRole("button", { name: "Takeback (1)" }).click();
  await expect(page.getByText("Start a game and make your move.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Takeback (0)" })).toBeDisabled();
});

test("hint shows the best move on the board panel", async ({ page }) => {
  await mockPlayOpponents(page);
  await page.route(`${playApiUrl}/new`, async (route) => {
    await fulfillJson(route, playState());
  });
  await page.route(
    (url) => url.href.startsWith(`${playApiUrl}/hint?`),
    async (route) => {
      await fulfillJson(route, {
        schema_version: "play-hint.v1",
        game_id: "mock-game",
        best_move: { uci: "e2e4", san: "e4" },
        from_square: "e2",
        to_square: "e4",
        promotion: null,
        hints_remaining: 2,
      });
    },
  );

  await page.goto("/play");
  await page.getByRole("button", { name: "Start game" }).click();
  await page.getByRole("button", { name: "Hint (3)" }).click();

  await expect(page.getByTestId("play-hint")).toContainText("Hint: e4");
  await expect(page.getByRole("button", { name: "Hint (2)" })).toBeVisible();
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

type MockPlayMove = {
  ply: number;
  side: "white" | "black";
  san: string;
  uci: string;
};

type MockMoveRef = {
  uci: string;
  san: string;
};

type MockLegalMoveGroup = {
  from_square: string;
  destinations: { to_square: string; promotions: string[] }[];
};

type MockPlayState = {
  schema_version: "play-state.v1";
  game_id: string;
  opponent: {
    kind: "maia" | "stockfish";
    requested: "auto" | "maia" | "stockfish";
    label: string;
    engine: string;
    maia_rating: 1100 | 1500 | 1900 | null;
    fallback_reason: string | null;
  };
  user_color: "white" | "black";
  status: "active" | "completed" | "resigned";
  result: "1-0" | "0-1" | "1/2-1/2" | "*";
  fen: string;
  orientation: "white" | "black";
  legal_moves: MockLegalMoveGroup[];
  moves: MockPlayMove[];
  bot_move: MockMoveRef | null;
  hints_remaining: number;
  takebacks_remaining: number;
  pgn: string | null;
};

async function mockPlayOpponents(page: Page): Promise<void> {
  await page.route(`${playApiUrl}/opponents`, async (route) => {
    await fulfillJson(route, {
      schema_version: "play-opponents.v1",
      default_requested: "stockfish",
      default_maia_rating: 1500,
      stockfish_path: "/opt/homebrew/bin/stockfish",
      stockfish_available: true,
      stockfish_label: "Stockfish fallback",
      maia: {
        lc0_path: null,
        lc0_available: false,
        weights_dir: "checkpoints/maia",
        ratings: [1100, 1500, 1900],
        available_ratings: [],
        missing_weights: [1100, 1500, 1900],
      },
    });
  });
}

async function fulfillJson(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function playState(overrides: Partial<MockPlayState> = {}): MockPlayState {
  return {
    schema_version: "play-state.v1",
    game_id: "mock-game",
    opponent: {
      kind: "stockfish",
      requested: "stockfish",
      label: "Stockfish fallback",
      engine: "Stockfish",
      maia_rating: null,
      fallback_reason: null,
    },
    user_color: "white",
    status: "active",
    result: "*",
    fen: startFen,
    orientation: "white",
    legal_moves: [
      {
        from_square: "e2",
        destinations: [{ to_square: "e4", promotions: [] }],
      },
    ],
    moves: [],
    bot_move: null,
    hints_remaining: 3,
    takebacks_remaining: 1,
    pgn: null,
    ...overrides,
  };
}

async function movePiece(
  page: Page,
  board: Locator,
  fromSquare: string,
  toSquare: string,
  orientation: "white" | "black" = "white",
): Promise<void> {
  const box = await board.boundingBox();
  if (!box) {
    throw new Error("Chessboard was not visible.");
  }
  const from = squareCenter(box, fromSquare, orientation);
  const to = squareCenter(box, toSquare, orientation);
  await page.mouse.click(from.x, from.y);
  await page.mouse.click(to.x, to.y);
}

async function dragPiece(
  page: Page,
  board: Locator,
  fromSquare: string,
  toSquare: string,
  orientation: "white" | "black" = "white",
): Promise<void> {
  const box = await board.boundingBox();
  if (!box) {
    throw new Error("Chessboard was not visible.");
  }
  const from = squareCenter(box, fromSquare, orientation);
  const to = squareCenter(box, toSquare, orientation);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 12 });
  await page.mouse.up();
}

function squareCenter(
  box: { x: number; y: number; width: number; height: number },
  square: string,
  orientation: "white" | "black" = "white",
): { x: number; y: number } {
  const file = square.charCodeAt(0) - "a".charCodeAt(0);
  const rank = Number(square[1]);
  if (orientation === "black") {
    return {
      x: box.x + ((7 - file + 0.5) * box.width) / 8,
      y: box.y + ((rank - 1 + 0.5) * box.height) / 8,
    };
  }
  return {
    x: box.x + ((file + 0.5) * box.width) / 8,
    y: box.y + ((8 - rank + 0.5) * box.height) / 8,
  };
}

function firstLegalMove(fen: string): { from: string; to: string } {
  const chess = new Chess(fen);
  const [move] = chess.moves({ verbose: true });
  if (!move) {
    throw new Error("Training drill had no legal moves.");
  }
  return { from: move.from, to: move.to };
}

function sideToMove(fen: string): "white" | "black" {
  return fen.split(" ")[1] === "b" ? "black" : "white";
}

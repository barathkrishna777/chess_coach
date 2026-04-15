"use client";

import { FormEvent, useMemo, useState } from "react";
import type { Key, KeyPair } from "chessground/types";

import Board from "@/components/Board";
import HealthIndicator from "@/components/HealthIndicator";

const API_URL = "http://localhost:8000/api/games";
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

type Score =
  | { type: "cp"; cp: number }
  | { type: "mate"; mate_in: number; winner: "white" | "black" };

type EngineMove = {
  uci: string;
  san: string;
};

type EngineAnalysis = {
  status: "ok" | "terminal";
  depth: number | null;
  score: Score;
  best_move: EngineMove | null;
  pv: EngineMove[];
  nodes: number | null;
  time_ms: number;
};

type AnnotatedMove = {
  ply: number;
  move_number: number;
  side: "white" | "black";
  san: string;
  uci: string;
  from_square: string;
  to_square: string;
  promotion: "q" | "r" | "b" | "n" | null;
  fen_before: string;
  fen_after: string;
  analysis_before: EngineAnalysis;
  analysis_after: EngineAnalysis;
  eval_delta_cp_white: number | null;
  loss_cp: number | null;
  is_engine_best: boolean;
};

type AnnotatedGame = {
  schema_version: "annotated-game.v1";
  game_id: string;
  headers: Record<string, string>;
  players: {
    white: { name: string | null; elo: number | null };
    black: { name: string | null; elo: number | null };
  };
  result: "1-0" | "0-1" | "1/2-1/2" | "*";
  initial_fen: string;
  final_fen: string;
  analysis: {
    engine: string;
    depth: number;
    positions_evaluated: number;
    elapsed_ms: number;
  };
  moves: AnnotatedMove[];
};

type ApiErrorEnvelope = {
  error?: {
    code?: string;
    message?: string;
  };
};

export default function Home() {
  const [pgn, setPgn] = useState("");
  const [game, setGame] = useState<AnnotatedGame | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedMove =
    game && selectedIndex >= 0 ? game.moves[selectedIndex] ?? null : null;
  const boardFen = selectedMove?.fen_after ?? game?.initial_fen ?? START_FEN;
  const currentScore =
    selectedMove?.analysis_after.score ??
    game?.moves[0]?.analysis_before.score ??
    ({ type: "cp", cp: 0 } satisfies Score);
  const lastMove = selectedMove ? lastMoveKeys(selectedMove) : null;

  const title = useMemo(() => gameTitle(game), [game]);

  async function analyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!pgn.trim()) {
      setError("Paste a PGN before starting analysis.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pgn }),
      });
      const body: unknown = await response.json();
      if (!response.ok) {
        throw new Error(errorMessage(body, response.status));
      }

      const annotatedGame = body as AnnotatedGame;
      setGame(annotatedGame);
      setSelectedIndex(annotatedGame.moves.length > 0 ? 0 : -1);
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function readFile(file: File) {
    setError(null);
    setPgn(await file.text());
  }

  return (
    <main className="min-h-screen bg-[#f6f8fb] text-[#17201d]">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-5 py-6 lg:px-8">
        <header className="flex flex-col gap-3 border-b border-[#d5ddd8] pb-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm font-medium uppercase tracking-wide text-[#37786f]">
              chess_ml
            </p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight">
              {title}
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-[#4a5a54]">
              Paste one PGN, then step through the game with Stockfish scores.
            </p>
          </div>
          <HealthIndicator />
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(360px,560px)_minmax(320px,1fr)] lg:items-start">
          <div className="flex gap-3">
            <EvalBar score={currentScore} />
            <div className="w-full max-w-[560px]">
              <Board fen={boardFen} lastMove={lastMove} />
            </div>
          </div>

          <div className="flex flex-col gap-5">
            <form
              onSubmit={analyze}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                const file = event.dataTransfer.files.item(0);
                if (file) void readFile(file);
              }}
              className="rounded-md border border-dashed border-[#8aa79c] bg-white p-4"
            >
              <label
                htmlFor="pgn"
                className="text-sm font-semibold text-[#17201d]"
              >
                PGN
              </label>
              <textarea
                id="pgn"
                value={pgn}
                onChange={(event) => setPgn(event.target.value)}
                placeholder={'[White "You"]\n[Black "Opponent"]\n\n1. e4 e5 2. Nf3 Nc6'}
                className="mt-2 h-40 w-full resize-y rounded-md border border-[#ccd6d1] bg-[#fbfcfd] p-3 font-mono text-sm leading-5 text-[#17201d] outline-none transition focus:border-[#37786f] focus:ring-2 focus:ring-[#cce8df]"
              />
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <label className="inline-flex cursor-pointer items-center rounded-md border border-[#ccd6d1] bg-white px-3 py-2 text-sm font-medium text-[#17201d] transition hover:border-[#37786f]">
                  Choose PGN
                  <input
                    type="file"
                    accept=".pgn,.txt"
                    className="sr-only"
                    onChange={(event) => {
                      const file = event.target.files?.item(0);
                      if (file) void readFile(file);
                    }}
                  />
                </label>
                <button
                  type="submit"
                  disabled={isAnalyzing}
                  className="rounded-md bg-[#d84f45] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#bd4138] disabled:cursor-not-allowed disabled:bg-[#a9b6b0]"
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze game"}
                </button>
                {game ? (
                  <span className="text-sm text-[#4a5a54]">
                    {game.analysis.positions_evaluated} positions,{" "}
                    {(game.analysis.elapsed_ms / 1000).toFixed(1)}s
                  </span>
                ) : null}
              </div>
              {error ? (
                <p className="mt-3 rounded-md bg-[#ffe4df] px-3 py-2 text-sm text-[#912f28]">
                  {error}
                </p>
              ) : null}
            </form>

            {game ? (
              <div className="grid gap-5 xl:grid-cols-[minmax(260px,1fr)_minmax(260px,0.9fr)]">
                <MoveList
                  moves={game.moves}
                  selectedIndex={selectedIndex}
                  onSelect={setSelectedIndex}
                  onStart={() => setSelectedIndex(-1)}
                />
                <CurrentMovePanel move={selectedMove} game={game} />
              </div>
            ) : (
              <div className="rounded-md border border-[#d5ddd8] bg-white p-4">
                <p className="text-sm leading-6 text-[#4a5a54]">
                  Stockfish is ready for one standard game at a time.
                </p>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

function MoveList({
  moves,
  selectedIndex,
  onSelect,
  onStart,
}: {
  moves: AnnotatedMove[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  onStart: () => void;
}) {
  return (
    <section className="rounded-md border border-[#d5ddd8] bg-white">
      <div className="flex items-center justify-between border-b border-[#e3e9e5] px-4 py-3">
        <h2 className="text-sm font-semibold">Moves</h2>
        <button
          type="button"
          onClick={onStart}
          className={`rounded-md px-2 py-1 text-xs font-medium ${
            selectedIndex === -1
              ? "bg-[#37786f] text-white"
              : "text-[#4a5a54] hover:bg-[#edf4f1]"
          }`}
        >
          Start
        </button>
      </div>
      <div className="max-h-[420px] overflow-y-auto p-2">
        {moves.map((move, index) => (
          <button
            key={`${move.ply}-${move.uci}`}
            type="button"
            onClick={() => onSelect(index)}
            className={`grid w-full grid-cols-[3rem_1fr_4.5rem] items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition ${
              selectedIndex === index
                ? "bg-[#e1f2ed] text-[#17201d]"
                : "hover:bg-[#f0f5f2]"
            }`}
          >
            <span className="font-mono text-xs text-[#65766f]">
              {move.side === "white" ? `${move.move_number}.` : `${move.move_number}...`}
            </span>
            <span className="font-semibold">{move.san}</span>
            <span className={`text-right text-xs ${lossClass(move.loss_cp)}`}>
              {lossLabel(move.loss_cp)}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

function CurrentMovePanel({
  move,
  game,
}: {
  move: AnnotatedMove | null;
  game: AnnotatedGame;
}) {
  if (!move) {
    return (
      <section className="rounded-md border border-[#d5ddd8] bg-white p-4">
        <h2 className="text-sm font-semibold">Start position</h2>
        <p className="mt-3 text-sm text-[#4a5a54]">
          {playerName(game.players.white)} vs {playerName(game.players.black)}
        </p>
      </section>
    );
  }

  return (
    <section className="rounded-md border border-[#d5ddd8] bg-white p-4">
      <h2 className="text-sm font-semibold">
        {move.move_number}
        {move.side === "white" ? "." : "..."} {move.san}
      </h2>
      <dl className="mt-4 grid gap-3 text-sm">
        <InfoRow label="After" value={scoreLabel(move.analysis_after.score)} />
        <InfoRow label="Before" value={scoreLabel(move.analysis_before.score)} />
        <InfoRow label="Loss" value={lossLabel(move.loss_cp)} />
        <InfoRow
          label="Best"
          value={move.analysis_before.best_move?.san ?? "Game over"}
        />
        <InfoRow
          label="Line"
          value={move.analysis_before.pv.map((pvMove) => pvMove.san).join(" ") || "None"}
        />
      </dl>
    </section>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[4rem_1fr] gap-3 border-b border-[#e3e9e5] pb-2 last:border-b-0 last:pb-0">
      <dt className="text-[#65766f]">{label}</dt>
      <dd className="min-w-0 break-words font-medium text-[#17201d]">{value}</dd>
    </div>
  );
}

function EvalBar({ score }: { score: Score }) {
  const whitePercent = scoreToWhitePercent(score);
  return (
    <div className="flex h-full min-h-[320px] w-8 shrink-0 flex-col overflow-hidden rounded-md border border-[#1f2a24] bg-[#2a2f2c]">
      <div
        className="mt-auto bg-[#f4f7f5] transition-[height] duration-300"
        style={{ height: `${whitePercent}%` }}
      />
    </div>
  );
}

function scoreToWhitePercent(score: Score): number {
  if (score.type === "mate") {
    return score.winner === "white" ? 100 : 0;
  }
  const clamped = Math.max(-800, Math.min(800, score.cp));
  return Math.round(((clamped + 800) / 1600) * 100);
}

function scoreLabel(score: Score): string {
  if (score.type === "mate") {
    const sign = score.winner === "white" ? "+" : "-";
    return `${sign}M${score.mate_in}`;
  }
  const pawns = score.cp / 100;
  return `${pawns >= 0 ? "+" : ""}${pawns.toFixed(2)}`;
}

function lossLabel(loss: number | null): string {
  if (loss === null) return "-";
  if (loss === 0) return "best";
  return `-${(loss / 100).toFixed(2)}`;
}

function lossClass(loss: number | null): string {
  if (loss === null || loss < 50) return "text-[#37786f]";
  if (loss < 150) return "text-[#9a6b16]";
  return "text-[#bd4138]";
}

function gameTitle(game: AnnotatedGame | null): string {
  if (!game) return "Review a game";
  return `${playerName(game.players.white)} vs ${playerName(game.players.black)}`;
}

function playerName(player: { name: string | null; elo: number | null }): string {
  const name = player.name?.trim() || "Unknown";
  return player.elo ? `${name} (${player.elo})` : name;
}

function errorMessage(body: unknown, status: number): string {
  if (isApiErrorEnvelope(body) && body.error?.message) {
    return body.error.message;
  }
  return `Analysis failed with HTTP ${status}.`;
}

function isApiErrorEnvelope(value: unknown): value is ApiErrorEnvelope {
  if (!value || typeof value !== "object") return false;
  const maybeEnvelope = value as { error?: unknown };
  if (!maybeEnvelope.error || typeof maybeEnvelope.error !== "object") return false;
  return true;
}

function lastMoveKeys(move: AnnotatedMove): KeyPair | null {
  const from = squareKey(move.from_square);
  const to = squareKey(move.to_square);
  if (!from || !to) return null;
  return [from, to];
}

function squareKey(square: string): Key | null {
  return /^[a-h][1-8]$/.test(square) ? (square as Key) : null;
}

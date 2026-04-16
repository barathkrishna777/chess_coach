"use client";

import { FormEvent, useMemo, useState } from "react";
import Link from "next/link";

import Board from "@/components/Board";
import GameReview, { gameTitle } from "@/components/GameReview";
import HealthIndicator from "@/components/HealthIndicator";
import { analyzePgn, userFacingErrorMessage } from "@/lib/api";
import type { AnnotatedGame } from "@/lib/types";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export default function Home() {
  const [pgn, setPgn] = useState("");
  const [game, setGame] = useState<AnnotatedGame | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      setGame(await analyzePgn(pgn));
    } catch (caught: unknown) {
      setError(userFacingErrorMessage(caught));
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
          <div className="flex flex-wrap items-center gap-3">
            <Link
              href="/play"
              className="rounded-md border border-[#37786f] px-3 py-2 text-sm font-semibold text-[#2c625a] transition hover:bg-[#edf4f1]"
            >
              Play a game
            </Link>
            <Link
              href="/dashboard"
              className="rounded-md border border-[#37786f] px-3 py-2 text-sm font-semibold text-[#2c625a] transition hover:bg-[#edf4f1]"
            >
              Dashboard
            </Link>
            <HealthIndicator />
          </div>
        </header>

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
          <label htmlFor="pgn" className="text-sm font-semibold text-[#17201d]">
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
            <p
              role="alert"
              aria-live="polite"
              className="mt-3 rounded-md bg-[#ffe4df] px-3 py-2 text-sm text-[#912f28]"
            >
              {error}
            </p>
          ) : null}
        </form>

        {game ? (
          <GameReview game={game} onGameChange={setGame} />
        ) : (
          <section className="grid gap-6 lg:grid-cols-[minmax(360px,560px)_minmax(320px,1fr)] lg:items-start">
            <div className="w-full max-w-[560px]">
              <Board fen={START_FEN} />
            </div>
            <div className="rounded-md border border-[#d5ddd8] bg-white p-4">
              <p className="text-sm leading-6 text-[#4a5a54]">
                Stockfish is ready for one standard game at a time.
              </p>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

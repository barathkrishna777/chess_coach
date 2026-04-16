"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { Dests, Key, KeyPair } from "chessground/types";

import Board from "@/components/Board";
import GameReview from "@/components/GameReview";
import HealthIndicator from "@/components/HealthIndicator";
import {
  analyzePgn,
  getPlayOpponents,
  resignPlayGame,
  startPlayGame,
  submitPlayMove,
} from "@/lib/api";
import type {
  AnnotatedGame,
  LegalMoveGroup,
  MaiaRating,
  PlayMove,
  PlayOpponentRequest,
  PlayOpponentsStatus,
  PlayState,
  PromotionChoice,
} from "@/lib/types";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export default function PlayPage() {
  const [playState, setPlayState] = useState<PlayState | null>(null);
  const [reviewGame, setReviewGame] = useState<AnnotatedGame | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isSubmittingMove, setIsSubmittingMove] = useState(false);
  const [isReviewing, setIsReviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opponentStatus, setOpponentStatus] = useState<PlayOpponentsStatus | null>(null);
  const [selectedOpponent, setSelectedOpponent] = useState<PlayOpponentRequest>("auto");
  const [selectedMaiaRating, setSelectedMaiaRating] = useState<MaiaRating>(1500);

  useEffect(() => {
    let cancelled = false;
    getPlayOpponents()
      .then((status) => {
        if (!cancelled) {
          setOpponentStatus(status);
          setSelectedOpponent(status.default_requested);
          setSelectedMaiaRating(status.default_maia_rating);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setOpponentStatus(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const legalDests = useMemo(
    () => legalDestsFromState(playState),
    [playState],
  );
  const lastMove = playState?.moves.length
    ? lastMoveKeys(playState.moves[playState.moves.length - 1])
    : null;
  const canMove =
    Boolean(playState) &&
    playState?.status === "active" &&
    !isSubmittingMove &&
    !isReviewing &&
    !reviewGame;

  async function startGame() {
    setIsStarting(true);
    setError(null);
    setReviewGame(null);
    try {
      setPlayState(
        await startPlayGame({
          opponent: selectedOpponent,
          maiaRating: selectedMaiaRating,
        }),
      );
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setIsStarting(false);
    }
  }

  async function submitMove(from: Key, to: Key) {
    if (!playState || playState.status !== "active" || isSubmittingMove) return;

    const uci = uciForMove(playState.legal_moves, from, to);
    if (!uci) {
      setError("Choose a legal move from the highlighted destinations.");
      return;
    }

    setIsSubmittingMove(true);
    setError(null);
    try {
      const nextState = await submitPlayMove(playState.game_id, uci);
      setPlayState(nextState);
      if (nextState.status !== "active") {
        await reviewFinishedGame(nextState);
      }
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setIsSubmittingMove(false);
    }
  }

  async function resign() {
    if (!playState || playState.status !== "active") return;
    setIsSubmittingMove(true);
    setError(null);
    try {
      const nextState = await resignPlayGame(playState.game_id);
      setPlayState(nextState);
      await reviewFinishedGame(nextState);
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setIsSubmittingMove(false);
    }
  }

  async function reviewFinishedGame(nextState: PlayState) {
    if (!nextState.pgn) {
      setError("Play at least one move before reviewing the game.");
      return;
    }

    setIsReviewing(true);
    try {
      setReviewGame(await analyzePgn(nextState.pgn));
    } finally {
      setIsReviewing(false);
    }
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
              Play, then review
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-[#4a5a54]">
              Play White against Maia when local setup is ready, then go straight
              into Stockfish-grounded review.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Link
              href="/"
              className="rounded-md border border-[#37786f] px-3 py-2 text-sm font-semibold text-[#2c625a] transition hover:bg-[#edf4f1]"
            >
              Review PGN
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

        {reviewGame ? (
          <GameReview game={reviewGame} onGameChange={setReviewGame} />
        ) : (
          <section className="grid gap-6 lg:grid-cols-[minmax(360px,560px)_minmax(320px,1fr)] lg:items-start">
            <div className="w-full max-w-[560px]">
              <Board
                fen={playState?.fen ?? START_FEN}
                lastMove={lastMove}
                legalDests={legalDests}
                disabled={!canMove}
                onMove={(from, to) => void submitMove(from, to)}
              />
            </div>

            <div className="flex flex-col gap-5">
              <section className="rounded-md border border-[#d5ddd8] bg-white p-4">
                <div className="flex flex-wrap items-center gap-3">
                  <label className="grid gap-1 text-sm">
                    <span className="font-semibold text-[#17201d]">Opponent</span>
                    <select
                      value={selectedOpponent}
                      disabled={Boolean(playState?.status === "active") || isStarting}
                      onChange={(event) =>
                        setSelectedOpponent(event.target.value as PlayOpponentRequest)
                      }
                      className="rounded-md border border-[#ccd6d1] bg-white px-3 py-2 text-sm outline-none transition focus:border-[#37786f] focus:ring-2 focus:ring-[#cce8df]"
                    >
                      <option value="auto">Auto: Maia with fallback</option>
                      <option value="maia">Maia only</option>
                      <option value="stockfish">Stockfish fallback</option>
                    </select>
                  </label>
                  <label className="grid gap-1 text-sm">
                    <span className="font-semibold text-[#17201d]">Maia rating</span>
                    <select
                      value={selectedMaiaRating}
                      disabled={
                        selectedOpponent === "stockfish" ||
                        Boolean(playState?.status === "active") ||
                        isStarting
                      }
                      onChange={(event) =>
                        setSelectedMaiaRating(Number(event.target.value) as MaiaRating)
                      }
                      className="rounded-md border border-[#ccd6d1] bg-white px-3 py-2 text-sm outline-none transition focus:border-[#37786f] focus:ring-2 focus:ring-[#cce8df] disabled:cursor-not-allowed disabled:bg-[#edf1ee]"
                    >
                      <option value={1100}>1100</option>
                      <option value={1500}>1500</option>
                      <option value={1900}>1900</option>
                    </select>
                  </label>
                  <button
                    type="button"
                    onClick={() => void startGame()}
                    disabled={isStarting || isSubmittingMove || isReviewing}
                    className="rounded-md bg-[#37786f] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#2c625a] disabled:cursor-not-allowed disabled:bg-[#a9b6b0]"
                  >
                    {playState ? "New game" : isStarting ? "Starting..." : "Start game"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void resign()}
                    disabled={
                      !playState ||
                      playState.status !== "active" ||
                      playState.moves.length === 0 ||
                      isSubmittingMove ||
                      isReviewing
                    }
                    className="rounded-md border border-[#d84f45] px-4 py-2 text-sm font-semibold text-[#bd4138] transition hover:bg-[#ffe4df] disabled:cursor-not-allowed disabled:border-[#ccd6d1] disabled:text-[#8a9992]"
                  >
                    Resign and review
                  </button>
                </div>

                <dl className="mt-4 grid gap-3 text-sm">
                  <InfoRow label="Status" value={statusLabel(playState)} />
                  <InfoRow
                    label="Opponent"
                    value={playState?.opponent.label ?? opponentSetupLabel(opponentStatus)}
                  />
                  <InfoRow
                    label="Setup"
                    value={opponentSetupDetail(opponentStatus, selectedMaiaRating)}
                  />
                  <InfoRow
                    label="Turn"
                    value={canMove ? "White to move" : turnLabel(playState, isReviewing)}
                  />
                </dl>

                {playState?.opponent.fallback_reason ? (
                  <p className="mt-4 rounded-md bg-[#fff2bf] px-3 py-2 text-sm text-[#6f4b00]">
                    Maia was not available, so this game started with Stockfish fallback:{" "}
                    {playState.opponent.fallback_reason}
                  </p>
                ) : null}

                {error ? (
                  <p className="mt-4 rounded-md bg-[#ffe4df] px-3 py-2 text-sm text-[#912f28]">
                    {error}
                  </p>
                ) : null}
              </section>

              <section className="rounded-md border border-[#d5ddd8] bg-white">
                <div className="border-b border-[#e3e9e5] px-4 py-3">
                  <h2 className="text-sm font-semibold">Game moves</h2>
                </div>
                <div className="max-h-[420px] overflow-y-auto p-2">
                  {playState?.moves.length ? (
                    playState.moves.map((move) => (
                      <div
                        key={`${move.ply}-${move.uci}`}
                        className="grid grid-cols-[3rem_minmax(0,1fr)] gap-2 rounded-md px-2 py-2 text-sm"
                      >
                        <span className="font-mono text-xs text-[#65766f]">
                          {move.side === "white"
                            ? `${Math.ceil(move.ply / 2)}.`
                            : `${Math.ceil(move.ply / 2)}...`}
                        </span>
                        <span className="font-semibold">{move.san}</span>
                      </div>
                    ))
                  ) : (
                    <p className="px-2 py-2 text-sm leading-6 text-[#4a5a54]">
                      Start a game and play White.
                    </p>
                  )}
                </div>
              </section>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[5rem_1fr] gap-3 border-b border-[#e3e9e5] pb-2 last:border-b-0 last:pb-0">
      <dt className="text-[#65766f]">{label}</dt>
      <dd className="min-w-0 break-words font-medium text-[#17201d]">{value}</dd>
    </div>
  );
}

function legalDestsFromState(state: PlayState | null): Dests {
  const dests: Dests = new Map();
  if (!state || state.status !== "active") return dests;

  for (const group of state.legal_moves) {
    const from = squareKey(group.from_square);
    if (!from) continue;
    const destinations = group.destinations
      .map((destination) => squareKey(destination.to_square))
      .filter((destination): destination is Key => destination !== null);
    dests.set(from, destinations);
  }
  return dests;
}

function uciForMove(
  legalMoves: LegalMoveGroup[],
  from: Key,
  to: Key,
): string | null {
  const group = legalMoves.find((candidate) => candidate.from_square === from);
  const destination = group?.destinations.find(
    (candidate) => candidate.to_square === to,
  );
  if (!destination) return null;
  if (destination.promotions.length === 0) return `${from}${to}`;

  const choice = promotionChoice(destination.promotions);
  return choice ? `${from}${to}${choice}` : null;
}

function promotionChoice(options: PromotionChoice[]): PromotionChoice | null {
  if (options.length === 1) return options[0];
  const answer = window
    .prompt("Promote to queen, rook, bishop, or knight.", "q")
    ?.trim()
    .toLowerCase();
  if (answer === "queen") return "q";
  if (answer === "rook") return "r";
  if (answer === "bishop") return "b";
  if (answer === "knight") return "n";
  return options.includes(answer as PromotionChoice)
    ? (answer as PromotionChoice)
    : null;
}

function statusLabel(state: PlayState | null): string {
  if (!state) return "Ready";
  if (state.status === "active") return "In progress";
  if (state.status === "resigned") return "Resigned";
  return `Completed ${state.result}`;
}

function turnLabel(state: PlayState | null, isReviewing: boolean): string {
  if (isReviewing) return "Preparing review";
  if (!state) return "Start when ready";
  if (state.status !== "active") return "Game over";
  return `Waiting for ${state.opponent.label}`;
}

function opponentSetupLabel(status: PlayOpponentsStatus | null): string {
  if (!status) return "Checking local opponents";
  if (status.maia.lc0_available && status.maia.available_ratings.length > 0) {
    return "Maia, with Stockfish fallback";
  }
  if (status.stockfish_available) return status.stockfish_label;
  return "No local opponent ready";
}

function opponentSetupDetail(
  status: PlayOpponentsStatus | null,
  selectedRating: MaiaRating,
): string {
  if (!status) return "Loading setup status";
  if (!status.stockfish_available && !status.maia.lc0_available) {
    return "Install Stockfish or Lc0 to play locally";
  }
  if (!status.maia.lc0_available) {
    return "Lc0 missing; auto mode will use Stockfish";
  }
  if (!status.maia.available_ratings.includes(selectedRating)) {
    return `Maia ${selectedRating} weights missing; auto mode will use Stockfish`;
  }
  return `Maia ${selectedRating} is ready`;
}

function lastMoveKeys(move: PlayMove): KeyPair | null {
  const from = squareKey(move.uci.slice(0, 2));
  const to = squareKey(move.uci.slice(2, 4));
  if (!from || !to) return null;
  return [from, to];
}

function squareKey(square: string): Key | null {
  return /^[a-h][1-8]$/.test(square) ? (square as Key) : null;
}

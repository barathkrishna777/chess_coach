"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import Link from "next/link";
import type { DrawShape } from "chessground/draw";
import type { Dests, Key, KeyPair } from "chessground/types";

import Board from "@/components/Board";
import GameReview from "@/components/GameReview";
import HealthIndicator from "@/components/HealthIndicator";
import {
  analyzePgn,
  getPlayHint,
  getPlayOpponents,
  resignPlayGame,
  startPlayGame,
  submitPlayMove,
  takebackPlayGame,
  userFacingErrorMessage,
} from "@/lib/api";
import { applyUciMoves } from "@/lib/chess";
import type {
  AnnotatedGame,
  LegalMoveDestination,
  LegalMoveGroup,
  MaiaRating,
  PlayColor,
  PlayHint,
  PlayMove,
  PlayOpponentRequest,
  PlayOpponentsStatus,
  PlayState,
  PromotionChoice,
} from "@/lib/types";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

type PendingPromotion = {
  from: Key;
  to: Key;
  options: PromotionChoice[];
};

const PROMOTION_LABELS: Record<PromotionChoice, string> = {
  q: "Queen",
  r: "Rook",
  b: "Bishop",
  n: "Knight",
};

export default function PlayPage() {
  const [playState, setPlayState] = useState<PlayState | null>(null);
  const [reviewGame, setReviewGame] = useState<AnnotatedGame | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isSubmittingMove, setIsSubmittingMove] = useState(false);
  const [isHinting, setIsHinting] = useState(false);
  const [isReviewing, setIsReviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opponentStatus, setOpponentStatus] = useState<PlayOpponentsStatus | null>(null);
  const [selectedOpponent, setSelectedOpponent] = useState<PlayOpponentRequest>("auto");
  const [selectedMaiaRating, setSelectedMaiaRating] = useState<MaiaRating>(1500);
  const [selectedUserColor, setSelectedUserColor] = useState<PlayColor>("white");
  const [pendingPromotion, setPendingPromotion] = useState<PendingPromotion | null>(null);
  const [hint, setHint] = useState<PlayHint | null>(null);
  const [boardResetNonce, setBoardResetNonce] = useState(0);
  const [viewedPly, setViewedPly] = useState(0);

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

  const currentPly = playState?.moves.length ?? 0;
  const isViewingCurrentPosition = viewedPly === currentPly;
  const viewedFens = useMemo(
    () => applyUciMoves(START_FEN, playState?.moves.map((move) => move.uci) ?? []),
    [playState?.moves],
  );
  const boardFen = viewedFens[viewedPly] ?? playState?.fen ?? START_FEN;
  const legalDests = useMemo(
    () => (isViewingCurrentPosition ? legalDestsFromState(playState) : new Map()),
    [isViewingCurrentPosition, playState],
  );
  const viewedLastMove = viewedPly > 0 ? (playState?.moves[viewedPly - 1] ?? null) : null;
  const lastMove = viewedLastMove
    ? lastMoveKeys(viewedLastMove)
    : null;
  const hasUserMove =
    playState?.moves.some((move) => move.side === playState.user_color) ?? false;
  const activeUserColor = playState?.user_color ?? selectedUserColor;
  const canMove =
    Boolean(playState) &&
    playState?.status === "active" &&
    playState.legal_moves.length > 0 &&
    isViewingCurrentPosition &&
    !isSubmittingMove &&
    !isReviewing &&
    !reviewGame;
  const canTakeback =
    Boolean(playState) &&
    playState?.status === "active" &&
    hasUserMove &&
    (playState?.takebacks_remaining ?? 0) > 0 &&
    !isSubmittingMove &&
    !isReviewing;
  const canHint =
    canMove &&
    (playState?.hints_remaining ?? 0) > 0 &&
    !isHinting &&
    !pendingPromotion;
  const hintShapes = useMemo<DrawShape[]>(() => {
    if (!hint) return [];
    const from = squareKey(hint.from_square);
    const to = squareKey(hint.to_square);
    if (!from || !to) return [];
    return [{ orig: from, dest: to, brush: "green" }];
  }, [hint]);

  useEffect(() => {
    if (viewedPly > currentPly) {
      setViewedPly(currentPly);
    }
  }, [currentPly, viewedPly]);

  useEffect(() => {
    if (reviewGame) return;

    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLSelectElement ||
        target instanceof HTMLTextAreaElement ||
        target.isContentEditable
      ) {
        return;
      }
      if (pendingPromotion) return;

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setViewedPly((ply) => Math.max(0, ply - 1));
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setViewedPly((ply) => Math.min(currentPly, ply + 1));
      }
    }

    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [currentPly, pendingPromotion, reviewGame]);

  async function startGame() {
    setIsStarting(true);
    setError(null);
    setReviewGame(null);
    setPendingPromotion(null);
    setHint(null);
    setBoardResetNonce((value) => value + 1);
    try {
      const nextState = await startPlayGame({
        opponent: selectedOpponent,
        maiaRating: selectedMaiaRating,
        userColor: selectedUserColor,
      });
      setPlayState(nextState);
      setViewedPly(nextState.moves.length);
    } catch (caught: unknown) {
      setPlayState(null);
      setError(userFacingErrorMessage(caught));
    } finally {
      setIsStarting(false);
    }
  }

  async function submitMove(from: Key, to: Key) {
    if (!playState || playState.status !== "active" || isSubmittingMove) return;

    const destination = legalDestinationForMove(playState.legal_moves, from, to);
    if (!destination) {
      setError("Choose a legal move from the highlighted destinations.");
      resetBoardToServerState();
      return;
    }

    if (destination.promotions.length > 0) {
      setError(null);
      setPendingPromotion({ from, to, options: destination.promotions });
      return;
    }

    await submitUci(`${from}${to}`);
  }

  async function submitPromotion(choice: PromotionChoice) {
    if (!pendingPromotion) return;
    const { from, to } = pendingPromotion;
    setPendingPromotion(null);
    await submitUci(`${from}${to}${choice}`);
  }

  function cancelPromotion() {
    setPendingPromotion(null);
    resetBoardToServerState();
  }

  async function submitUci(uci: string) {
    if (!playState) return;
    setIsSubmittingMove(true);
    setError(null);
    setHint(null);
    try {
      const nextState = await submitPlayMove(playState.game_id, uci);
      setPlayState(nextState);
      if (nextState.bot_move || nextState.status !== "active") {
        setViewedPly(nextState.moves.length);
      }
      if (nextState.status !== "active") {
        await reviewFinishedGame(nextState);
      }
    } catch (caught: unknown) {
      setError(userFacingErrorMessage(caught));
      resetBoardToServerState();
    } finally {
      setIsSubmittingMove(false);
    }
  }

  async function takeback() {
    if (!playState || playState.status !== "active") return;
    setIsSubmittingMove(true);
    setError(null);
    setHint(null);
    setPendingPromotion(null);
    try {
      const nextState = await takebackPlayGame(playState.game_id);
      setPlayState(nextState);
      setViewedPly(nextState.moves.length);
      resetBoardToServerState();
    } catch (caught: unknown) {
      setError(userFacingErrorMessage(caught));
    } finally {
      setIsSubmittingMove(false);
    }
  }

  async function requestHint() {
    if (!playState || !canHint) return;
    setIsHinting(true);
    setError(null);
    try {
      const nextHint = await getPlayHint(playState.game_id);
      setHint(nextHint);
      setPlayState((current) =>
        current && current.game_id === nextHint.game_id
          ? { ...current, hints_remaining: nextHint.hints_remaining }
          : current,
      );
    } catch (caught: unknown) {
      setError(userFacingErrorMessage(caught));
    } finally {
      setIsHinting(false);
    }
  }

  async function resign() {
    if (!playState || playState.status !== "active") return;
    setIsSubmittingMove(true);
    setError(null);
    setHint(null);
    setPendingPromotion(null);
    try {
      const nextState = await resignPlayGame(playState.game_id);
      setPlayState(nextState);
      setViewedPly(nextState.moves.length);
      await reviewFinishedGame(nextState);
    } catch (caught: unknown) {
      setError(userFacingErrorMessage(caught));
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
    } catch (caught: unknown) {
      setError(userFacingErrorMessage(caught));
    } finally {
      setIsReviewing(false);
    }
  }

  function resetBoardToServerState() {
    setBoardResetNonce((value) => value + 1);
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
              Play either side against Maia when local setup is ready, then go
              straight into Stockfish-grounded review.
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
          <section className="flex flex-col gap-4">
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-[#d5ddd8] bg-white p-4">
              <div>
                <p className="text-sm font-semibold text-[#17201d]">Post-game review</p>
                <p className="mt-1 text-sm text-[#4a5a54]">
                  Start another game with the same opponent, rating, and color.
                </p>
              </div>
              <button
                type="button"
                onClick={() => void startGame()}
                disabled={isStarting || isSubmittingMove || isReviewing}
                className="rounded-md bg-[#37786f] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#2c625a] disabled:cursor-not-allowed disabled:bg-[#a9b6b0]"
              >
                {isStarting ? "Starting..." : "Play again"}
              </button>
            </div>
            {error ? (
              <p
                role="alert"
                aria-live="polite"
                className="rounded-md bg-[#ffe4df] px-3 py-2 text-sm text-[#912f28]"
              >
                {error}
              </p>
            ) : null}
            <GameReview game={reviewGame} onGameChange={setReviewGame} />
          </section>
        ) : (
          <section className="grid gap-6 lg:grid-cols-[minmax(360px,560px)_minmax(320px,1fr)] lg:items-start">
            <div className="relative w-full max-w-[560px]">
              <Board
                key={`${playState?.game_id ?? "start"}-${boardResetNonce}`}
                fen={boardFen}
                orientation={playState?.orientation ?? selectedUserColor}
                turnColor={activeUserColor}
                movableColor={activeUserColor}
                lastMove={lastMove}
                legalDests={legalDests}
                shapes={isViewingCurrentPosition ? hintShapes : []}
                disabled={!canMove || Boolean(pendingPromotion)}
                onMove={(from, to) => void submitMove(from, to)}
              />
              {pendingPromotion ? (
                <PromotionDialog
                  pending={pendingPromotion}
                  orientation={playState?.orientation ?? selectedUserColor}
                  onSelect={(choice) => void submitPromotion(choice)}
                  onCancel={cancelPromotion}
                />
              ) : null}
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
                  <label className="grid gap-1 text-sm">
                    <span className="font-semibold text-[#17201d]">Play as</span>
                    <select
                      value={selectedUserColor}
                      disabled={Boolean(playState?.status === "active") || isStarting}
                      onChange={(event) => setSelectedUserColor(event.target.value as PlayColor)}
                      className="rounded-md border border-[#ccd6d1] bg-white px-3 py-2 text-sm outline-none transition focus:border-[#37786f] focus:ring-2 focus:ring-[#cce8df]"
                    >
                      <option value="white">White</option>
                      <option value="black">Black</option>
                    </select>
                  </label>
                  <button
                    type="button"
                    onClick={() => void startGame()}
                    disabled={isStarting || isSubmittingMove || isReviewing}
                    className="rounded-md bg-[#37786f] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#2c625a] disabled:cursor-not-allowed disabled:bg-[#a9b6b0]"
                  >
                    {isStarting ? "Starting..." : playState ? "New game" : "Start game"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void requestHint()}
                    disabled={!canHint}
                    className="rounded-md border border-[#37786f] px-4 py-2 text-sm font-semibold text-[#2c625a] transition hover:bg-[#edf4f1] disabled:cursor-not-allowed disabled:border-[#ccd6d1] disabled:text-[#8a9992]"
                  >
                    {isHinting ? "Finding..." : `Hint (${playState?.hints_remaining ?? 3})`}
                  </button>
                  <button
                    type="button"
                    onClick={() => void takeback()}
                    disabled={!canTakeback}
                    className="rounded-md border border-[#37786f] px-4 py-2 text-sm font-semibold text-[#2c625a] transition hover:bg-[#edf4f1] disabled:cursor-not-allowed disabled:border-[#ccd6d1] disabled:text-[#8a9992]"
                  >
                    Takeback ({playState?.takebacks_remaining ?? 1})
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
                  <InfoRow label="Color" value={colorLabel(activeUserColor)} />
                  <InfoRow
                    label="Setup"
                    value={opponentSetupDetail(opponentStatus, selectedMaiaRating)}
                  />
                  <InfoRow
                    label="Turn"
                    value={
                      !isViewingCurrentPosition && playState?.status === "active"
                        ? "Use → to return to the live board"
                        : canMove
                          ? `${colorLabel(activeUserColor)} to move`
                          : turnLabel(playState, isReviewing)
                    }
                  />
                  <InfoRow
                    label="Board"
                    value={boardPositionLabel(playState, viewedPly, currentPly)}
                  />
                </dl>

                {playState?.status === "active" && !isViewingCurrentPosition ? (
                  <p className="mt-4 rounded-md bg-[#fff2bf] px-3 py-2 text-sm text-[#6f4b00]">
                    You are looking at an earlier position. Press → until you reach the
                    live board before making your next move.
                  </p>
                ) : null}

                {hint ? (
                  <p
                    data-testid="play-hint"
                    className="mt-4 rounded-md bg-[#edf4f1] px-3 py-2 text-sm text-[#2c625a]"
                  >
                    Hint: {hint.best_move.san}
                  </p>
                ) : null}

                {playState?.opponent.fallback_reason ? (
                  <p className="mt-4 rounded-md bg-[#fff2bf] px-3 py-2 text-sm text-[#6f4b00]">
                    Maia was not available, so this game started with Stockfish fallback:{" "}
                    {playState.opponent.fallback_reason}
                  </p>
                ) : null}

                {error ? (
                  <p
                    role="alert"
                    aria-live="polite"
                    className="mt-4 rounded-md bg-[#ffe4df] px-3 py-2 text-sm text-[#912f28]"
                  >
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
                      Start a game and make your move.
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

function PromotionDialog({
  pending,
  orientation,
  onSelect,
  onCancel,
}: {
  pending: PendingPromotion;
  orientation: PlayColor;
  onSelect: (choice: PromotionChoice) => void;
  onCancel: () => void;
}) {
  return (
    <div
      className="absolute z-10 rounded-md border border-[#1f2a24] bg-white p-2 shadow-lg"
      style={promotionDialogStyle(pending.to, orientation)}
      role="dialog"
      aria-label="Choose promotion piece"
    >
      <div className="grid grid-cols-2 gap-2">
        {pending.options.map((choice) => (
          <button
            key={choice}
            type="button"
            onClick={() => onSelect(choice)}
            className="rounded-md border border-[#ccd6d1] px-3 py-2 text-sm font-semibold text-[#17201d] transition hover:bg-[#edf4f1]"
            aria-label={`Promote to ${PROMOTION_LABELS[choice]}`}
          >
            {PROMOTION_LABELS[choice]}
          </button>
        ))}
      </div>
      <button
        type="button"
        onClick={onCancel}
        className="mt-2 w-full rounded-md border border-[#ccd6d1] px-3 py-1.5 text-xs font-semibold text-[#4a5a54] transition hover:bg-[#f6f8fb]"
      >
        Cancel
      </button>
    </div>
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

function legalDestinationForMove(
  legalMoves: LegalMoveGroup[],
  from: Key,
  to: Key,
): LegalMoveDestination | null {
  const group = legalMoves.find((candidate) => candidate.from_square === from);
  return group?.destinations.find((candidate) => candidate.to_square === to) ?? null;
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

function boardPositionLabel(
  state: PlayState | null,
  viewedPly: number,
  currentPly: number,
): string {
  if (!state) return "Start position";
  if (viewedPly === currentPly) return "Live position";
  if (viewedPly === 0) return "Start position";
  const move = state.moves[viewedPly - 1];
  if (!move) return "Earlier position";
  const moveNumber = Math.ceil(move.ply / 2);
  const prefix = move.side === "white" ? `${moveNumber}.` : `${moveNumber}...`;
  return `After ${prefix} ${move.san}`;
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

function promotionDialogStyle(square: string, orientation: PlayColor): CSSProperties {
  const center = squareCenterPercent(square, orientation);
  return {
    left: `${center.x}%`,
    top: `${center.y}%`,
    transform: "translate(-50%, -50%)",
  };
}

function squareCenterPercent(
  square: string,
  orientation: PlayColor,
): { x: number; y: number } {
  if (!/^[a-h][1-8]$/.test(square)) {
    return { x: 50, y: 50 };
  }
  const file = square.charCodeAt(0) - "a".charCodeAt(0);
  const rank = Number(square[1]);
  if (orientation === "black") {
    return {
      x: ((7 - file + 0.5) * 100) / 8,
      y: ((rank - 1 + 0.5) * 100) / 8,
    };
  }
  return {
    x: ((file + 0.5) * 100) / 8,
    y: ((8 - rank + 0.5) * 100) / 8,
  };
}

function colorLabel(color: PlayColor): string {
  return color === "white" ? "White" : "Black";
}

function squareKey(square: string): Key | null {
  return /^[a-h][1-8]$/.test(square) ? (square as Key) : null;
}

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Dests, Key, KeyPair } from "chessground/types";

import Board from "@/components/Board";
import { explainMove, getExplanationStatus, userFacingErrorMessage } from "@/lib/api";
import {
  applyUciMoves,
  legalDestsFromFen,
  sideToMoveFromFen,
  uciFromTo,
  uciToKeyPair,
} from "@/lib/chess";
import type {
  AnnotatedGame,
  AnnotatedMove,
  EngineMove,
  ExplanationStatus,
  Motif,
  MotifSeverity,
  MoveExplanation,
  Score,
} from "@/lib/types";

type GameReviewProps = {
  game: AnnotatedGame;
  onGameChange: (game: AnnotatedGame) => void;
};

type TryResult = "correct" | "incorrect";
type MoveListEntry = { move: AnnotatedMove; index: number };
type MovePair = {
  moveNumber: number;
  white: MoveListEntry | null;
  black: MoveListEntry | null;
};

export default function GameReview({ game, onGameChange }: GameReviewProps) {
  const [selectedIndex, setSelectedIndex] = useState(game.moves.length > 0 ? 0 : -1);
  const [explainingPly, setExplainingPly] = useState<number | null>(null);
  const [coachStatus, setCoachStatus] = useState<ExplanationStatus | null>(null);
  const [coachStatusError, setCoachStatusError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // PV mode: null = off, 0..N = current step index into pvData.fens
  const [pvStep, setPvStep] = useState<number | null>(null);
  // Try mode
  const [tryMode, setTryMode] = useState(false);
  const [tryResult, setTryResult] = useState<TryResult | null>(null);

  const pvMode = pvStep !== null;

  useEffect(() => {
    let cancelled = false;
    getExplanationStatus()
      .then((status) => {
        if (!cancelled) {
          setCoachStatus(status);
          setCoachStatusError(null);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setCoachStatusError(caught instanceof Error ? caught.message : String(caught));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setSelectedIndex(game.moves.length > 0 ? 0 : -1);
    setError(null);
  }, [game.game_id, game.moves.length]);

  // Exit interactive modes when selection changes
  useEffect(() => {
    setPvStep(null);
    setTryMode(false);
    setTryResult(null);
  }, [selectedIndex]);

  const selectedMove = selectedIndex >= 0 ? (game.moves[selectedIndex] ?? null) : null;

  // Compute PV positions (fen_before + each PV move applied)
  const pvData = useMemo(() => {
    if (!selectedMove) return null;
    const ucis = selectedMove.analysis_before.pv.map((m) => m.uci);
    if (ucis.length === 0) return null;
    const fens = applyUciMoves(selectedMove.fen_before, ucis);
    return { ucis, fens };
  }, [selectedMove]);

  // Try mode: legal dests computed from fen_before when try mode is active
  const tryLegalDests = useMemo<Dests | undefined>(() => {
    if (!tryMode || !selectedMove) return undefined;
    return legalDestsFromFen(selectedMove.fen_before);
  }, [tryMode, selectedMove]);

  // Keyboard navigation
  const selectedIndexRef = useRef(selectedIndex);
  selectedIndexRef.current = selectedIndex;
  const pvStepRef = useRef(pvStep);
  pvStepRef.current = pvStep;
  const pvDataRef = useRef(pvData);
  pvDataRef.current = pvData;
  const tryModeRef = useRef(tryMode);
  tryModeRef.current = tryMode;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target.isContentEditable
      ) {
        return;
      }

      const step = pvStepRef.current;
      const inPv = step !== null;
      const data = pvDataRef.current;

      if (inPv) {
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          setPvStep((s) => (s !== null ? Math.max(0, s - 1) : null));
        } else if (e.key === "ArrowRight") {
          e.preventDefault();
          if (data) {
            setPvStep((s) => (s !== null ? Math.min(data.fens.length - 1, s + 1) : null));
          }
        } else if (e.key === "Escape") {
          e.preventDefault();
          setPvStep(null);
        }
        return;
      }

      if (tryModeRef.current) return;

      const moves = game.moves;
      const idx = selectedIndexRef.current;

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setSelectedIndex((i) => Math.max(0, i - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setSelectedIndex((i) => {
          if (moves.length === 0) return -1;
          return i >= moves.length - 1 ? 0 : i + 1;
        });
      } else if (e.key === "ArrowUp") {
        const prev = findPrevFlagged(moves, idx);
        if (prev !== -1) {
          e.preventDefault();
          setSelectedIndex(prev);
        }
      } else if (e.key === "ArrowDown") {
        const next = findNextFlagged(moves, idx);
        if (next !== -1) {
          e.preventDefault();
          setSelectedIndex(next);
        }
      }
    }

    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [game.moves]);

  // Board state depending on mode
  let boardFen: string;
  let boardLastMove: KeyPair | null;
  let boardOnMove: ((from: Key, to: Key) => void) | undefined;
  let boardLegalDests: Dests | undefined;
  let boardTurnColor: "white" | "black" | undefined;
  let boardMovableColor: "white" | "black" | undefined;

  if (pvMode && pvData) {
    const step = pvStep ?? 1;
    boardFen = pvData.fens[step] ?? selectedMove?.fen_before ?? game.initial_fen;
    const prevUci = step > 0 ? pvData.ucis[step - 1] : null;
    boardLastMove = prevUci ? uciToKeyPair(prevUci) : null;
  } else if (tryMode && selectedMove) {
    boardFen = selectedMove.fen_before;
    boardLastMove = null;
    if (!tryResult) {
      boardLegalDests = tryLegalDests;
      boardTurnColor = sideToMoveFromFen(selectedMove.fen_before);
      boardMovableColor = sideToMoveFromFen(selectedMove.fen_before);
      boardOnMove = (from, to) => {
        const played = from + to;
        const best = selectedMove.analysis_before.best_move?.uci;
        const correct = best !== undefined && uciFromTo(best) === played;
        setTryResult(correct ? "correct" : "incorrect");
      };
    }
  } else {
    boardFen = selectedMove?.fen_after ?? game.initial_fen;
    boardLastMove = selectedMove ? lastMoveKeys(selectedMove) : null;
  }

  const currentScore =
    selectedMove?.analysis_after.score ??
    game.moves[0]?.analysis_before.score ??
    ({ type: "cp", cp: 0 } satisfies Score);

  async function explainSelectedMove() {
    if (!selectedMove || selectedMove.motifs.length === 0) return;

    setExplainingPly(selectedMove.ply);
    setError(null);
    try {
      const explanation = await explainMove(
        selectedMove,
        actualLineFromSelection(game.moves, selectedIndex),
      );
      onGameChange({
        ...game,
        moves: game.moves.map((move) =>
          move.ply === selectedMove.ply ? { ...move, explanation } : move,
        ),
      });
    } catch (caught: unknown) {
      setError(userFacingErrorMessage(caught));
    } finally {
      setExplainingPly(null);
    }
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[minmax(360px,560px)_minmax(320px,1fr)] lg:items-start">
      <div className="flex flex-col gap-2">
        <div className="flex gap-3">
          <EvalBar score={currentScore} />
          <div className="w-full max-w-[560px]">
            <Board
              fen={boardFen}
              lastMove={boardLastMove}
              onMove={boardOnMove}
              legalDests={boardLegalDests}
              turnColor={boardTurnColor ?? "white"}
              movableColor={boardMovableColor}
            />
          </div>
        </div>

      {/* Main line navigation bar */}
        {pvMode && pvData ? (
          <div className="flex items-center gap-2 rounded-md border border-[#c2d8d0] bg-[#edf4f1] px-3 py-2">
            <button
              type="button"
              onClick={() => setPvStep((s) => (s !== null ? Math.max(0, s - 1) : null))}
              disabled={(pvStep ?? 0) === 0}
              className="rounded px-2 py-1 text-sm font-semibold text-[#2c625a] disabled:opacity-30 hover:bg-[#d5ece5]"
              title="Previous (←)"
            >
              ←
            </button>
            <span className="flex-1 text-center text-xs text-[#4a5a54]">
              Main line — step {pvStep ?? 0} of {pvData.fens.length - 1}
            </span>
            <button
              type="button"
              onClick={() =>
                setPvStep((s) =>
                  s !== null ? Math.min(pvData.fens.length - 1, s + 1) : null,
                )
              }
              disabled={(pvStep ?? 0) >= pvData.fens.length - 1}
              className="rounded px-2 py-1 text-sm font-semibold text-[#2c625a] disabled:opacity-30 hover:bg-[#d5ece5]"
              title="Next (→)"
            >
              →
            </button>
            <button
              type="button"
              onClick={() => setPvStep(null)}
              className="rounded px-2 py-1 text-xs font-medium text-[#4a5a54] hover:bg-[#d5ece5]"
              title="Exit (Esc)"
            >
              ✕ Back to game
            </button>
          </div>
        ) : null}

        {/* Try mode bar */}
        {tryMode && selectedMove ? (
          <div
            className={`flex items-center gap-3 rounded-md border px-3 py-2 ${
              tryResult === "correct"
                ? "border-[#37786f] bg-[#edf4f1]"
                : tryResult === "incorrect"
                  ? "border-[#e28a82] bg-[#ffe4df]"
                  : "border-[#c2d8d0] bg-[#edf4f1]"
            }`}
          >
            {tryResult === null ? (
              <span className="flex-1 text-xs text-[#4a5a54]">
                Find the best move from this position (you are{" "}
                {sideToMoveFromFen(selectedMove.fen_before)}).
              </span>
            ) : tryResult === "correct" ? (
              <span className="flex-1 text-xs font-semibold text-[#2c625a]">
                Correct! That&apos;s the engine&apos;s best move.
              </span>
            ) : (
              <span className="flex-1 text-xs text-[#912f28]">
                Not quite.{" "}
                {selectedMove.analysis_before.best_move
                  ? `The engine played ${selectedMove.analysis_before.best_move.san}.`
                  : "The engine found a better move."}
              </span>
            )}
            <button
              type="button"
              onClick={() => {
                setTryMode(false);
                setTryResult(null);
              }}
              className="rounded px-2 py-1 text-xs font-medium text-[#4a5a54] hover:bg-[#d5ece5]"
            >
              ✕ Back to review
            </button>
          </div>
        ) : null}
      </div>

      <div className="flex flex-col gap-5">
        {error ? (
          <p
            role="alert"
            aria-live="polite"
            className="rounded-md bg-[#ffe4df] px-3 py-2 text-sm text-[#912f28]"
          >
            {error}
          </p>
        ) : null}

        {game.opening ? <OpeningSummary opening={game.opening} /> : null}
        <ReviewSummary moves={game.moves} />

        <div className="grid gap-5 xl:grid-cols-[minmax(260px,1fr)_minmax(260px,0.9fr)]">
          <MoveList
            moves={game.moves}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
            onStart={() => setSelectedIndex(-1)}
          />
          <CurrentMovePanel
            move={selectedMove}
            game={game}
            coachStatus={coachStatus}
            coachStatusError={coachStatusError}
            isExplaining={selectedMove?.ply === explainingPly}
            pvData={pvData}
            pvMode={pvMode}
            tryMode={tryMode}
            onExplain={() => void explainSelectedMove()}
            onShowBestLine={() => setPvStep(1)}
            onTryEnginesMove={() => {
              setPvStep(null);
              setTryMode(true);
              setTryResult(null);
            }}
          />
        </div>

        <p className="text-xs text-[#8aa79c]">
          Keyboard: ← → to step moves · ↑ ↓ to jump between flagged moves
        </p>
      </div>
    </section>
  );
}

function OpeningSummary({ opening }: { opening: { eco: string; name: string } }) {
  return (
    <div className="rounded-md border border-[#d5ddd8] bg-white px-4 py-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-[#65766f]">
        Opening
      </p>
      <p className="mt-1 text-sm font-medium text-[#17201d]">
        {opening.eco} {"—"} {opening.name}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Review summary card
// ---------------------------------------------------------------------------

function ReviewSummary({ moves }: { moves: AnnotatedMove[] }) {
  const summary = useMemo(() => {
    if (moves.length === 0) return null;

    const accuracy = Math.round(
      (moves.filter((m) => (m.loss_cp ?? 0) < 50).length / moves.length) * 100,
    );

    const top2Blunders = moves
      .filter((m) => m.motifs.length > 0 && m.loss_cp !== null)
      .sort((a, b) => (b.loss_cp ?? 0) - (a.loss_cp ?? 0))
      .slice(0, 2);

    const motifCounts = new Map<string, { label: string; count: number }>();
    for (const move of moves) {
      for (const motif of move.motifs) {
        const entry = motifCounts.get(motif.id) ?? { label: motif.label, count: 0 };
        entry.count += 1;
        motifCounts.set(motif.id, entry);
      }
    }
    const topMotif = [...motifCounts.values()].sort((a, b) => b.count - a.count)[0] ?? null;

    return { accuracy, top2Blunders, topMotif };
  }, [moves]);

  if (!summary) return null;

  return (
    <div className="rounded-md border border-[#d5ddd8] bg-white px-4 py-3">
      <h2 className="text-sm font-semibold">Review summary</h2>
      <div className="mt-3 flex flex-wrap gap-5 text-sm">
        <div>
          <p className="text-xs text-[#65766f]">Accuracy</p>
          <p
            className={`mt-0.5 text-lg font-semibold ${
              summary.accuracy >= 80
                ? "text-[#37786f]"
                : summary.accuracy >= 60
                  ? "text-[#9a6b16]"
                  : "text-[#bd4138]"
            }`}
          >
            {summary.accuracy}%
          </p>
        </div>
        {summary.topMotif ? (
          <div>
            <p className="text-xs text-[#65766f]">Most common mistake</p>
            <p className="mt-0.5 font-medium text-[#17201d]">{summary.topMotif.label}</p>
            <p className="text-xs text-[#65766f]">{summary.topMotif.count}×</p>
          </div>
        ) : null}
        {summary.top2Blunders.length > 0 ? (
          <div className="min-w-0">
            <p className="text-xs text-[#65766f]">Biggest mistakes</p>
            {summary.top2Blunders.map((move) => (
              <p key={move.ply} className="mt-0.5 text-xs text-[#17201d]">
                <span className="font-mono text-[#65766f]">
                  {move.move_number}
                  {move.side === "white" ? "." : "..."}{" "}
                </span>
                {move.san}
                <span className="ml-1 text-[#bd4138]">({lossLabel(move.loss_cp)})</span>
              </p>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Move list
// ---------------------------------------------------------------------------

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
  const pairs = movePairs(moves);

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
      <div className="grid grid-cols-[3rem_minmax(0,1fr)_minmax(0,1fr)] gap-2 border-b border-[#e3e9e5] px-4 py-2 text-xs font-semibold uppercase tracking-wide text-[#65766f]">
        <span>No.</span>
        <span>White</span>
        <span>Black</span>
      </div>
      <div className="max-h-[420px] overflow-y-auto p-2">
        {pairs.map((pair) => (
          <div
            key={pair.moveNumber}
            className="grid grid-cols-[3rem_minmax(0,1fr)_minmax(0,1fr)] gap-2 rounded-md px-2 py-1"
          >
            <span className="pt-2 font-mono text-xs text-[#65766f]">{pair.moveNumber}.</span>
            <MoveCell move={pair.white} selectedIndex={selectedIndex} onSelect={onSelect} />
            <MoveCell move={pair.black} selectedIndex={selectedIndex} onSelect={onSelect} />
          </div>
        ))}
      </div>
    </section>
  );
}

function MoveCell({
  move,
  selectedIndex,
  onSelect,
}: {
  move: MoveListEntry | null;
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  if (!move) return <span aria-hidden="true" />;

  return (
    <button
      type="button"
      onClick={() => onSelect(move.index)}
      className={`min-w-0 rounded-md px-2 py-2 text-left text-sm transition ${
        selectedIndex === move.index ? "bg-[#e1f2ed] text-[#17201d]" : "hover:bg-[#f0f5f2]"
      }`}
    >
      <span className="flex min-w-0 items-start justify-between gap-2">
        <span className="min-w-0">
          <span className="font-semibold">{move.move.san}</span>
          <MotifChips motifs={move.move.motifs} compact />
        </span>
        <span className={`shrink-0 text-xs ${lossClass(move.move.loss_cp)}`}>
          {lossLabel(move.move.loss_cp)}
        </span>
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Current move panel
// ---------------------------------------------------------------------------

function CurrentMovePanel({
  move,
  game,
  coachStatus,
  coachStatusError,
  isExplaining,
  pvData,
  pvMode,
  tryMode,
  onExplain,
  onShowBestLine,
  onTryEnginesMove,
}: {
  move: AnnotatedMove | null;
  game: AnnotatedGame;
  coachStatus: ExplanationStatus | null;
  coachStatusError: string | null;
  isExplaining: boolean;
  pvData: { ucis: string[]; fens: string[] } | null;
  pvMode: boolean;
  tryMode: boolean;
  onExplain: () => void;
  onShowBestLine: () => void;
  onTryEnginesMove: () => void;
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

  const hasBestLine = pvData !== null && pvData.ucis.length > 0;
  const hasBestMove = move.analysis_before.best_move !== null;

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
          label="Main line"
          value={move.analysis_before.pv.map((pvMove) => pvMove.san).join(" ") || "None"}
        />
      </dl>

      {/* Explore buttons */}
      {(hasBestLine || hasBestMove) && !pvMode && !tryMode ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {hasBestLine ? (
            <button
              type="button"
              onClick={onShowBestLine}
              className="rounded-md border border-[#37786f] px-3 py-1.5 text-xs font-semibold text-[#2c625a] transition hover:bg-[#edf4f1]"
            >
              Show main line
            </button>
          ) : null}
          {hasBestMove ? (
            <button
              type="button"
              onClick={onTryEnginesMove}
              className="rounded-md border border-[#37786f] px-3 py-1.5 text-xs font-semibold text-[#2c625a] transition hover:bg-[#edf4f1]"
            >
              Try engine&apos;s move
            </button>
          ) : null}
        </div>
      ) : null}

      {/* Mistake types */}
      <div className="mt-4 border-t border-[#e3e9e5] pt-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-[#65766f]">
          Mistake type
        </h3>
        {move.motifs.length > 0 ? (
          <>
            <MotifChips motifs={move.motifs} />
            <MotifEvidenceList motifs={move.motifs} />
          </>
        ) : (
          <p className="mt-2 text-sm text-[#4a5a54]">No clear mistake type detected.</p>
        )}
      </div>

      {/* Coach note */}
      <div className="mt-4 border-t border-[#e3e9e5] pt-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-[#65766f]">
          Coach
        </h3>
        <ExplanationText
          explanation={move.explanation}
          hasMotifs={move.motifs.length > 0}
          coachStatus={coachStatus}
          coachStatusError={coachStatusError}
          isExplaining={isExplaining}
          onExplain={onExplain}
        />
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Motif evidence details
// ---------------------------------------------------------------------------

function MotifEvidenceList({ motifs }: { motifs: Motif[] }) {
  const evidenced = motifs.filter(
    (m) =>
      m.evidence.piece !== null ||
      m.evidence.attackers.length > 0 ||
      m.evidence.defenders.length > 0 ||
      m.evidence.opponent_reply !== null,
  );
  if (evidenced.length === 0) return null;

  return (
    <ul className="mt-2 grid gap-1.5">
      {evidenced.map((motif) => (
        <li key={motif.id} className="text-xs leading-5 text-[#4a5a54]">
          {motifEvidenceSentence(motif)}
        </li>
      ))}
    </ul>
  );
}

function motifEvidenceSentence(motif: Motif): string {
  const { piece, attackers, defenders, opponent_reply, phase } = motif.evidence;
  const parts: string[] = [];

  if (piece) {
    const role = piece.role.charAt(0).toUpperCase() + piece.role.slice(1);
    parts.push(`${role} on ${piece.square}`);
  }
  if (attackers.length > 0) {
    parts.push(`attacked from ${attackers.join(", ")}`);
  }
  if (defenders.length > 0) {
    parts.push(`defended from ${defenders.join(", ")}`);
  } else if (attackers.length > 0) {
    parts.push("undefended");
  }
  if (opponent_reply) {
    parts.push(`opponent replied ${opponent_reply.san}`);
  }
  const phaseTag = phase !== "middlegame" ? ` (${phase})` : "";
  return parts.length > 0 ? parts.join(", ") + phaseTag + "." : "";
}

// ---------------------------------------------------------------------------
// Explanation text
// ---------------------------------------------------------------------------

function ExplanationText({
  explanation,
  hasMotifs,
  coachStatus,
  coachStatusError,
  isExplaining,
  onExplain,
}: {
  explanation: MoveExplanation | null;
  hasMotifs: boolean;
  coachStatus: ExplanationStatus | null;
  coachStatusError: string | null;
  isExplaining: boolean;
  onExplain: () => void;
}) {
  if (!hasMotifs) {
    return (
      <p className="mt-2 text-sm leading-6 text-[#4a5a54]">
        No coaching note for this move.
      </p>
    );
  }
  const configText = coachConfigText(coachStatus, coachStatusError);
  const requestBlocked = coachRequestBlocked(coachStatus);
  if (isExplaining) {
    return (
      <div className="mt-2 grid gap-2">
        <p className="text-sm leading-6 text-[#17201d]">
          Generating a grounded coach note from the Stockfish line. This can take a little
          while locally.
        </p>
        <p className="text-xs leading-5 text-[#65766f]">{configText}</p>
        <button
          type="button"
          disabled
          className="w-fit rounded-md bg-[#a9b6b0] px-3 py-2 text-sm font-semibold text-white"
        >
          Generating...
        </button>
      </div>
    );
  }
  if (!explanation) {
    return (
      <div className="mt-2 grid gap-2">
        <p className="text-sm leading-6 text-[#4a5a54]">No coach note requested yet.</p>
        <p className="text-xs leading-5 text-[#65766f]">{configText}</p>
        {requestBlocked ? (
          <p className="text-sm leading-6 text-[#912f28]">
            Local coach notes are not configured right now.
          </p>
        ) : (
          <button
            type="button"
            onClick={onExplain}
            className="w-fit rounded-md bg-[#37786f] px-3 py-2 text-sm font-semibold text-white transition hover:bg-[#2c625a]"
          >
            Generate coach note
          </button>
        )}
      </div>
    );
  }
  if (explanation.status === "ok" && explanation.text) {
    return (
      <div className="mt-2 grid gap-2">
        <p className="text-sm leading-6 text-[#17201d]">{explanation.text}</p>
        <p className="text-xs leading-5 text-[#65766f]">
          {explanationSourceLabel(explanation)}
        </p>
      </div>
    );
  }
  if (explanation.status === "unavailable") {
    return (
      <RetryableCoachMessage
        tone="neutral"
        message={unavailableMessage(explanation)}
        retryable={explanation.retryable}
        onExplain={onExplain}
        configText={configText}
      />
    );
  }
  return (
    <RetryableCoachMessage
      tone="error"
      message={errorExplanationMessage(explanation)}
      retryable={explanation.retryable}
      onExplain={onExplain}
      configText={configText}
    />
  );
}

function RetryableCoachMessage({
  tone,
  message,
  retryable,
  onExplain,
  configText,
}: {
  tone: "neutral" | "error";
  message: string;
  retryable: boolean;
  onExplain: () => void;
  configText: string;
}) {
  return (
    <div className="mt-2 grid gap-2">
      <p
        className={`text-sm leading-6 ${
          tone === "error" ? "text-[#912f28]" : "text-[#4a5a54]"
        }`}
      >
        {message}
      </p>
      <p className="text-xs leading-5 text-[#65766f]">{configText}</p>
      {retryable ? (
        <button
          type="button"
          onClick={onExplain}
          className="w-fit rounded-md border border-[#37786f] px-3 py-2 text-sm font-semibold text-[#2c625a] transition hover:bg-[#edf4f1]"
        >
          Retry coach note
        </button>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small helpers / pure components
// ---------------------------------------------------------------------------

function coachConfigText(
  status: ExplanationStatus | null,
  statusError: string | null,
): string {
  if (statusError) return "Coach config could not be loaded.";
  if (!status) return "Checking local coach config.";
  const timeout = `${trimSeconds(status.timeout_seconds)}s budget`;
  if (!status.enabled) return `Coach notes are disabled, ${timeout}.`;
  if (!status.configured) return `Coach provider is not configured, ${timeout}.`;
  return `${providerModelLabel(status.provider, status.model)}, ${timeout}.`;
}

function coachRequestBlocked(status: ExplanationStatus | null): boolean {
  if (!status) return false;
  return !status.enabled || !status.configured;
}

function providerModelLabel(
  provider: MoveExplanation["provider"],
  model: string | null,
): string {
  const providerLabel =
    provider === "ollama"
      ? "Ollama"
      : provider === "anthropic"
        ? "Anthropic"
        : provider === "codex"
          ? "Codex"
          : "local coach";
  return model ? `${providerLabel} ${model}` : providerLabel;
}

function explanationSourceLabel(explanation: MoveExplanation): string {
  if (explanation.source === "cache") return "Cached coach note.";
  if (explanation.source === "fallback") {
    return "Stockfish-grounded fallback. The model answer was rejected and not cached.";
  }
  return `Generated by ${providerModelLabel(explanation.provider, explanation.model)}.`;
}

function unavailableMessage(explanation: MoveExplanation): string {
  if (explanation.reason === "local_model_unavailable") {
    return "Ollama or the configured local model is unavailable. Start Ollama and pull the model, then retry.";
  }
  if (explanation.reason === "api_key_missing") {
    return "The selected coach provider is missing local configuration.";
  }
  return "Coach notes are unavailable right now.";
}

function errorExplanationMessage(explanation: MoveExplanation): string {
  if (explanation.reason === "timeout") {
    const timeout =
      explanation.timeout_seconds === null
        ? "the local timeout"
        : `${trimSeconds(explanation.timeout_seconds)}s`;
    return `The local coach hit the ${timeout} budget before returning a trusted note. Nothing was cached.`;
  }
  if (explanation.reason === "invalid_response") {
    return "The model responded, but the answer could not be trusted against Stockfish's line. Nothing was cached.";
  }
  if (explanation.reason === "provider_error") {
    return "The coach provider returned an error before a trusted note could be generated.";
  }
  return "The coaching note could not be grounded for this move.";
}

function trimSeconds(value: number): string {
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
}

function MotifChips({
  motifs,
  compact = false,
}: {
  motifs: Motif[];
  compact?: boolean;
}) {
  if (motifs.length === 0) return null;

  return (
    <span className={`flex flex-wrap gap-1 ${compact ? "mt-1" : "mt-2"}`}>
      {motifs.map((motif) => (
        <span
          key={motif.id}
          className={`rounded-md border px-2 py-0.5 text-[11px] font-semibold ${motifClass(
            motif.severity,
          )}`}
        >
          {motif.label}
        </span>
      ))}
    </span>
  );
}

function movePairs(moves: AnnotatedMove[]): MovePair[] {
  const pairs = new Map<number, MovePair>();

  moves.forEach((move, index) => {
    const pair = pairs.get(move.move_number) ?? {
      moveNumber: move.move_number,
      white: null,
      black: null,
    };
    pair[move.side] = { move, index };
    pairs.set(move.move_number, pair);
  });

  return [...pairs.values()].sort((a, b) => a.moveNumber - b.moveNumber);
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

function motifClass(severity: MotifSeverity): string {
  if (severity === "inaccuracy") {
    return "border-[#e1bc4f] bg-[#fff2bf] text-[#6f4b00]";
  }
  if (severity === "mistake") {
    return "border-[#e3a45d] bg-[#ffe4c7] text-[#7a3f10]";
  }
  return "border-[#e28a82] bg-[#ffe4df] text-[#912f28]";
}

export function gameTitle(game: AnnotatedGame | null): string {
  if (!game) return "Review a game";
  return `${playerName(game.players.white)} vs ${playerName(game.players.black)}`;
}

function playerName(player: { name: string | null; elo: number | null }): string {
  const name = player.name?.trim() || "Unknown";
  return player.elo ? `${name} (${player.elo})` : name;
}

function actualLineFromSelection(
  moves: AnnotatedMove[],
  selectedIndex: number,
): EngineMove[] {
  if (selectedIndex < 0) return [];
  return moves.slice(selectedIndex, selectedIndex + 6).map((move) => ({
    uci: move.uci,
    san: move.san,
  }));
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

function findNextFlagged(moves: AnnotatedMove[], currentIndex: number): number {
  for (let i = currentIndex + 1; i < moves.length; i++) {
    if ((moves[i]?.motifs.length ?? 0) > 0) return i;
  }
  return -1;
}

function findPrevFlagged(moves: AnnotatedMove[], currentIndex: number): number {
  for (let i = currentIndex - 1; i >= 0; i--) {
    if ((moves[i]?.motifs.length ?? 0) > 0) return i;
  }
  return -1;
}

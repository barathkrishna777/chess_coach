"use client";

import { useEffect, useMemo, useState } from "react";
import type { Key, KeyPair } from "chessground/types";

import Board from "@/components/Board";
import { explainMove } from "@/lib/api";
import type {
  AnnotatedGame,
  AnnotatedMove,
  EngineMove,
  Motif,
  MotifSeverity,
  MoveExplanation,
  Score,
} from "@/lib/types";

type GameReviewProps = {
  game: AnnotatedGame;
  onGameChange: (game: AnnotatedGame) => void;
};

export default function GameReview({ game, onGameChange }: GameReviewProps) {
  const [selectedIndex, setSelectedIndex] = useState(game.moves.length > 0 ? 0 : -1);
  const [explainingPly, setExplainingPly] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSelectedIndex(game.moves.length > 0 ? 0 : -1);
    setError(null);
  }, [game.game_id, game.moves.length]);

  const selectedMove =
    selectedIndex >= 0 ? game.moves[selectedIndex] ?? null : null;
  const boardFen = selectedMove?.fen_after ?? game.initial_fen;
  const currentScore =
    selectedMove?.analysis_after.score ??
    game.moves[0]?.analysis_before.score ??
    ({ type: "cp", cp: 0 } satisfies Score);
  const lastMove = selectedMove ? lastMoveKeys(selectedMove) : null;

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
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setExplainingPly(null);
    }
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[minmax(360px,560px)_minmax(320px,1fr)] lg:items-start">
      <div className="flex gap-3">
        <EvalBar score={currentScore} />
        <div className="w-full max-w-[560px]">
          <Board fen={boardFen} lastMove={lastMove} />
        </div>
      </div>

      <div className="flex flex-col gap-5">
        {error ? (
          <p className="rounded-md bg-[#ffe4df] px-3 py-2 text-sm text-[#912f28]">
            {error}
          </p>
        ) : null}
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
            isExplaining={selectedMove?.ply === explainingPly}
            onExplain={() => void explainSelectedMove()}
          />
        </div>
      </div>
    </section>
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
            className={`grid w-full grid-cols-[3rem_minmax(0,1fr)_4.5rem] items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition ${
              selectedIndex === index
                ? "bg-[#e1f2ed] text-[#17201d]"
                : "hover:bg-[#f0f5f2]"
            }`}
          >
            <span className="font-mono text-xs text-[#65766f]">
              {move.side === "white" ? `${move.move_number}.` : `${move.move_number}...`}
            </span>
            <span className="min-w-0">
              <span className="font-semibold">{move.san}</span>
              <MotifChips motifs={move.motifs} compact />
            </span>
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
  isExplaining,
  onExplain,
}: {
  move: AnnotatedMove | null;
  game: AnnotatedGame;
  isExplaining: boolean;
  onExplain: () => void;
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
      <div className="mt-4 border-t border-[#e3e9e5] pt-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-[#65766f]">
          Motifs
        </h3>
        {move.motifs.length > 0 ? (
          <MotifChips motifs={move.motifs} />
        ) : (
          <p className="mt-2 text-sm text-[#4a5a54]">No motif detected.</p>
        )}
      </div>
      <div className="mt-4 border-t border-[#e3e9e5] pt-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-[#65766f]">
          Coach
        </h3>
        <ExplanationText
          explanation={move.explanation}
          hasMotifs={move.motifs.length > 0}
          isExplaining={isExplaining}
          onExplain={onExplain}
        />
      </div>
    </section>
  );
}

function ExplanationText({
  explanation,
  hasMotifs,
  isExplaining,
  onExplain,
}: {
  explanation: MoveExplanation | null;
  hasMotifs: boolean;
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
  if (!explanation) {
    return (
      <button
        type="button"
        onClick={onExplain}
        disabled={isExplaining}
        className="mt-2 rounded-md bg-[#37786f] px-3 py-2 text-sm font-semibold text-white transition hover:bg-[#2c625a] disabled:cursor-not-allowed disabled:bg-[#a9b6b0]"
      >
        {isExplaining ? "Generating..." : "Generate coach note"}
      </button>
    );
  }
  if (explanation.status === "ok" && explanation.text) {
    return (
      <p className="mt-2 text-sm leading-6 text-[#17201d]">
        {explanation.text}
      </p>
    );
  }
  if (explanation.status === "unavailable") {
    return (
      <p className="mt-2 text-sm leading-6 text-[#4a5a54]">
        Start Ollama and pull the local explanation model to turn on coaching
        notes.
      </p>
    );
  }
  return (
    <p className="mt-2 text-sm leading-6 text-[#912f28]">
      The coaching note could not be grounded for this move.
    </p>
  );
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

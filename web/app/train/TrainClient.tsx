"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { Key } from "chessground/types";

import Board from "@/components/Board";
import {
  getNextDrill,
  submitDrillResult,
  userFacingErrorMessage,
} from "@/lib/api";
import {
  legalDestsFromFen,
  sideToMoveFromFen,
  uciForMoveFromFen,
} from "@/lib/chess";
import type { TrainingDrill, TrainingResult } from "@/lib/types";

export default function TrainClient() {
  const searchParams = useSearchParams();
  const motif = searchParams.get("motif") || "any";
  const [drill, setDrill] = useState<TrainingDrill | null>(null);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [attemptedUci, setAttemptedUci] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [emptyMessage, setEmptyMessage] = useState<string | null>(null);

  const loadDrill = useCallback(async () => {
    setIsLoading(true);
    setIsSubmitting(false);
    setResult(null);
    setAttemptedUci(null);
    setError(null);
    setEmptyMessage(null);
    try {
      const next = await getNextDrill(motif);
      setDrill(next);
    } catch (caught: unknown) {
      setDrill(null);
      const message = userFacingErrorMessage(caught);
      if (message.includes("No due drills")) {
        setEmptyMessage(message);
      } else {
        setError(message);
      }
    } finally {
      setIsLoading(false);
    }
  }, [motif]);

  useEffect(() => {
    void loadDrill();
  }, [loadDrill]);

  const legalDests = useMemo(() => {
    if (!drill || result) return undefined;
    return legalDestsFromFen(drill.fen);
  }, [drill, result]);

  const orientation = drill ? sideToMoveFromFen(drill.fen) : "white";

  async function handleMove(from: Key, to: Key) {
    if (!drill || result || isSubmitting) return;
    const uci = uciForMoveFromFen(drill.fen, from, to);
    setAttemptedUci(uci);
    setIsSubmitting(true);
    setError(null);
    try {
      const submitted = await submitDrillResult(drill.drill_id, uci);
      setResult(submitted);
    } catch (caught: unknown) {
      setError(userFacingErrorMessage(caught));
    } finally {
      setIsSubmitting(false);
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
              Personal drills
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-[#4a5a54]">
              Replay the moments your own games marked as worth fixing.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <NavLink href="/dashboard">Dashboard</NavLink>
            <NavLink href="/">Review PGN</NavLink>
            <NavLink href="/play">Play</NavLink>
          </div>
        </header>

        {error ? (
          <p
            role="alert"
            aria-live="polite"
            className="rounded-md bg-[#ffe4df] px-3 py-2 text-sm text-[#912f28]"
          >
            {error}
          </p>
        ) : null}

        {isLoading ? (
          <section className="rounded-md border border-[#d5ddd8] bg-white p-5">
            <p className="text-sm leading-6 text-[#4a5a54]">
              Loading the next due position.
            </p>
          </section>
        ) : drill ? (
          <section className="grid gap-6 lg:grid-cols-[minmax(360px,560px)_minmax(320px,1fr)] lg:items-start">
            <div className="w-full max-w-[560px]">
              <Board
                fen={drill.fen}
                orientation={orientation}
                turnColor={orientation}
                movableColor={orientation}
                legalDests={legalDests}
                disabled={Boolean(result) || isSubmitting}
                onMove={handleMove}
                testId="training-board"
              />
            </div>
            <aside className="flex flex-col gap-4">
              <section className="rounded-md border border-[#d5ddd8] bg-white p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-[#65766f]">
                      {drill.motif_label}
                    </p>
                    <h2 className="mt-1 text-lg font-semibold">
                      Move {drill.move_number}
                      {drill.side === "white" ? "." : "..."} Find the repair
                    </h2>
                  </div>
                  <span className="rounded-md bg-[#edf4f1] px-2 py-1 text-xs font-semibold text-[#2c625a]">
                    {orientation} to move
                  </span>
                </div>
                <p className="mt-3 text-sm leading-6 text-[#4a5a54]">
                  {drill.hint_text}
                </p>
                <p className="mt-2 text-xs leading-5 text-[#65766f]">
                  The answer stays hidden until you make a move.
                </p>
              </section>

              <FeedbackPanel
                result={result}
                attemptedUci={attemptedUci}
                isSubmitting={isSubmitting}
              />

              {result ? (
                <section className="rounded-md border border-[#d5ddd8] bg-white p-4">
                  <h2 className="text-sm font-semibold">Review the position</h2>
                  <dl className="mt-3 grid gap-2 text-sm">
                    <InfoRow label="Original move" value={result.context.played_move.san} />
                    <InfoRow
                      label="Loss"
                      value={
                        result.context.loss_cp === null
                          ? "Mate swing"
                          : `${result.context.loss_cp} cp`
                      }
                    />
                    <InfoRow label="Phase" value={result.context.phase} />
                    <InfoRow
                      label="Main line"
                      value={result.context.pv.map((move) => move.san).join(" ") || "None"}
                    />
                  </dl>
                  <EvidenceText evidence={result.context.evidence} />
                  {result.context.explanation_text ? (
                    <p className="mt-3 text-sm leading-6 text-[#17201d]">
                      {result.context.explanation_text}
                    </p>
                  ) : (
                    <p className="mt-3 text-sm leading-6 text-[#4a5a54]">
                      No stored coach note for this position yet.
                    </p>
                  )}
                  <button
                    type="button"
                    onClick={() => void loadDrill()}
                    className="mt-4 rounded-md bg-[#37786f] px-3 py-2 text-sm font-semibold text-white transition hover:bg-[#2c625a]"
                  >
                    Next puzzle
                  </button>
                </section>
              ) : null}
            </aside>
          </section>
        ) : (
          <EmptyState message={emptyMessage ?? "No drills are available yet."} />
        )}
      </div>
    </main>
  );
}

function FeedbackPanel({
  result,
  attemptedUci,
  isSubmitting,
}: {
  result: TrainingResult | null;
  attemptedUci: string | null;
  isSubmitting: boolean;
}) {
  if (isSubmitting) {
    return (
      <section className="rounded-md border border-[#c2d8d0] bg-[#edf4f1] p-4">
        <p className="text-sm font-semibold text-[#2c625a]">Checking your move.</p>
      </section>
    );
  }
  if (!result) {
    return (
      <section className="rounded-md border border-[#d5ddd8] bg-white p-4">
        <p className="text-sm leading-6 text-[#4a5a54]">
          Make one legal move on the board to get immediate feedback.
        </p>
      </section>
    );
  }

  return (
    <section
      className={`rounded-md border p-4 ${
        result.correct
          ? "border-[#37786f] bg-[#edf4f1]"
          : "border-[#e28a82] bg-[#ffe4df]"
      }`}
    >
      <p
        className={`text-sm font-semibold ${
          result.correct ? "text-[#2c625a]" : "text-[#912f28]"
        }`}
      >
        {result.correct ? "Correct" : "Incorrect"}
      </p>
      <p className="mt-2 text-sm leading-6 text-[#17201d]">
        You played <span className="font-mono">{attemptedUci ?? result.attempted_uci}</span>.
      </p>
      <p className="mt-1 text-sm leading-6 text-[#17201d]" data-testid="revealed-best-move">
        Best move:{" "}
        <span className="font-semibold">
          {result.best_move.san} ({result.best_move.uci})
        </span>
      </p>
    </section>
  );
}

function EvidenceText({ evidence }: { evidence: Record<string, unknown> | null }) {
  if (!evidence) return null;
  const piece = pieceLabel(evidence.piece);
  const attackers = stringList(evidence.attackers);
  const defenders = stringList(evidence.defenders);
  const opponentReply = moveLabel(evidence.opponent_reply);
  const parts = [
    piece,
    attackers.length > 0 ? `attacked from ${attackers.join(", ")}` : null,
    defenders.length > 0 ? `defended from ${defenders.join(", ")}` : null,
    opponentReply ? `opponent reply ${opponentReply}` : null,
  ].filter((part): part is string => Boolean(part));
  if (parts.length === 0) return null;
  return <p className="mt-3 text-xs leading-5 text-[#65766f]">{parts.join(", ")}.</p>;
}

function EmptyState({ message }: { message: string }) {
  return (
    <section className="rounded-md border border-[#d5ddd8] bg-white p-5">
      <h2 className="text-lg font-semibold">No due drills</h2>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-[#4a5a54]">{message}</p>
      <div className="mt-4 flex flex-wrap gap-3">
        <NavLink href="/dashboard">Back to dashboard</NavLink>
        <NavLink href="/">Review another game</NavLink>
      </div>
    </section>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[6rem_minmax(0,1fr)] gap-3">
      <dt className="text-xs font-semibold uppercase tracking-wide text-[#65766f]">
        {label}
      </dt>
      <dd className="min-w-0 text-[#17201d]">{value}</dd>
    </div>
  );
}

function NavLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link
      href={href}
      className="rounded-md border border-[#37786f] px-3 py-2 text-sm font-semibold text-[#2c625a] transition hover:bg-[#edf4f1]"
    >
      {children}
    </Link>
  );
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function pieceLabel(value: unknown): string | null {
  if (!isRecord(value)) return null;
  const role = value.role;
  const square = value.square;
  if (typeof role !== "string" || typeof square !== "string") return null;
  return `${role.charAt(0).toUpperCase()}${role.slice(1)} on ${square}`;
}

function moveLabel(value: unknown): string | null {
  if (!isRecord(value)) return null;
  const san = value.san;
  return typeof san === "string" ? san : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

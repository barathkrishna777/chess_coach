"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { ReactNode } from "react";

import HealthIndicator from "@/components/HealthIndicator";
import { getProfileDashboard } from "@/lib/api";
import type {
  ProfileDashboard,
  ProfileMotifAggregate,
  ProfilePhaseAggregate,
  RecentProfileGame,
} from "@/lib/types";

export default function DashboardPage() {
  const [profile, setProfile] = useState<ProfileDashboard | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getProfileDashboard()
      .then((dashboard) => {
        if (!cancelled) {
          setProfile(dashboard);
          setError(null);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const maxMotifCount = useMemo(
    () => Math.max(1, ...(profile?.motifs.map((motif) => motif.count) ?? [0])),
    [profile],
  );

  return (
    <main className="min-h-screen bg-[#f6f8fb] text-[#17201d]">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-5 py-6 lg:px-8">
        <header className="flex flex-col gap-3 border-b border-[#d5ddd8] pb-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm font-medium uppercase tracking-wide text-[#37786f]">
              chess_ml
            </p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight">
              Local profile
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-[#4a5a54]">
              Reviewed games become a running map of the mistakes worth training next.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <NavLink href="/">Review PGN</NavLink>
            <NavLink href="/play">Play</NavLink>
            <HealthIndicator />
          </div>
        </header>

        {error ? (
          <p className="rounded-md bg-[#ffe4df] px-3 py-2 text-sm text-[#912f28]">
            {error}
          </p>
        ) : null}

        {isLoading ? (
          <section className="rounded-md border border-[#d5ddd8] bg-white p-5">
            <p className="text-sm leading-6 text-[#4a5a54]">
              Loading local profile.
            </p>
          </section>
        ) : profile && profile.totals.games_reviewed > 0 ? (
          <>
            <SummaryGrid profile={profile} />
            <section className="grid gap-6 lg:grid-cols-[minmax(320px,1fr)_minmax(320px,0.85fr)]">
              <MotifList motifs={profile.motifs} maxCount={maxMotifCount} />
              <PhaseBreakdown phases={profile.phase_breakdown} />
            </section>
            <RecentGames games={profile.recent_games} />
          </>
        ) : (
          <EmptyState />
        )}
      </div>
    </main>
  );
}

function SummaryGrid({ profile }: { profile: ProfileDashboard }) {
  const totals = profile.totals;
  return (
    <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
      <SummaryTile label="Games" value={totals.games_reviewed.toString()} />
      <SummaryTile label="Moves" value={totals.moves_reviewed.toString()} />
      <SummaryTile label="Flagged" value={totals.flagged_moves.toString()} />
      <SummaryTile label="Motifs" value={totals.motif_occurrences.toString()} />
      <SummaryTile
        label="Per 100 moves"
        value={formatRate(totals.motif_rate_per_100_moves)}
      />
    </section>
  );
}

function SummaryTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-[#d5ddd8] bg-white p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-[#65766f]">
        {label}
      </p>
      <p className="mt-2 text-2xl font-semibold text-[#17201d]">{value}</p>
    </div>
  );
}

function MotifList({
  motifs,
  maxCount,
}: {
  motifs: ProfileMotifAggregate[];
  maxCount: number;
}) {
  return (
    <section className="rounded-md border border-[#d5ddd8] bg-white p-4">
      <h2 className="text-sm font-semibold">Motif profile</h2>
      {motifs.length > 0 ? (
        <div className="mt-4 grid gap-3">
          {motifs.map((motif) => (
            <div key={motif.id} className="grid gap-2">
              <div className="flex items-baseline justify-between gap-3">
                <p className="font-medium">{motif.label}</p>
                <p className="shrink-0 text-sm text-[#4a5a54]">
                  {motif.count} · {formatRate(motif.rate_per_100_moves)} / 100
                </p>
              </div>
              <div className="h-3 overflow-hidden rounded-md bg-[#edf1ee]">
                <div
                  className="h-full rounded-md bg-[#d84f45]"
                  style={{ width: `${Math.max(6, (motif.count / maxCount) * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-3 text-sm leading-6 text-[#4a5a54]">
          No motifs have been flagged yet.
        </p>
      )}
    </section>
  );
}

function PhaseBreakdown({ phases }: { phases: ProfilePhaseAggregate[] }) {
  return (
    <section className="rounded-md border border-[#d5ddd8] bg-white p-4">
      <h2 className="text-sm font-semibold">Phase breakdown</h2>
      <div className="mt-4 grid gap-3">
        {phases.map((phase) => (
          <div
            key={phase.phase}
            className="flex items-center justify-between gap-4 border-b border-[#e3e9e5] pb-3 last:border-b-0 last:pb-0"
          >
            <div>
              <p className="font-medium capitalize">{phase.phase}</p>
              <p className="text-sm text-[#65766f]">
                {formatRate(phase.rate_per_100_moves)} motifs / 100 moves
              </p>
            </div>
            <p className="text-lg font-semibold">{phase.count}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function RecentGames({ games }: { games: RecentProfileGame[] }) {
  return (
    <section className="rounded-md border border-[#d5ddd8] bg-white">
      <div className="border-b border-[#e3e9e5] px-4 py-3">
        <h2 className="text-sm font-semibold">Recent games</h2>
      </div>
      <div className="divide-y divide-[#e3e9e5]">
        {games.map((game) => (
          <div
            key={game.game_id}
            className="grid gap-3 px-4 py-3 md:grid-cols-[minmax(0,1fr)_8rem_8rem_8rem]"
          >
            <div className="min-w-0">
              <p className="truncate font-medium">
                {playerName(game.players.white)} vs {playerName(game.players.black)}
              </p>
              <p className="mt-1 text-xs text-[#65766f]">
                {sourceLabel(game.source)} · {formatDate(game.updated_at)}
              </p>
            </div>
            <Stat label="Result" value={game.result} />
            <Stat label="Moves" value={game.ply_count.toString()} />
            <Stat label="Flagged" value={game.flagged_moves.toString()} />
          </div>
        ))}
      </div>
    </section>
  );
}

function EmptyState() {
  return (
    <section className="rounded-md border border-[#d5ddd8] bg-white p-5">
      <h2 className="text-lg font-semibold">No reviewed games yet</h2>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-[#4a5a54]">
        Reviewed PGNs and just-played reviews will appear here after analysis.
        Start with one game, then come back to see which motifs are repeating.
      </p>
      <div className="mt-4 flex flex-wrap gap-3">
        <NavLink href="/">Review PGN</NavLink>
        <NavLink href="/play">Play a game</NavLink>
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide text-[#65766f]">
        {label}
      </p>
      <p className="mt-1 font-medium">{value}</p>
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

function playerName(player: { name: string | null; elo: number | null }): string {
  const name = player.name?.trim() || "Unknown";
  return player.elo ? `${name} (${player.elo})` : name;
}

function sourceLabel(source: RecentProfileGame["source"]): string {
  return source === "local_play" ? "Local play" : "PGN upload";
}

function formatRate(value: number): string {
  return value.toFixed(value % 1 === 0 ? 0 : 2);
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

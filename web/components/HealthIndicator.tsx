"use client";

import { useEffect, useState } from "react";

type Health = { status: string; version: string };

export default function HealthIndicator() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("http://localhost:8000/health")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<Health>;
      })
      .then((data) => {
        if (!cancelled) setHealth(data);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const color = error ? "bg-[#d84f45]" : health ? "bg-[#37786f]" : "bg-[#a9b6b0]";
  const label = error
    ? `API unreachable: ${error}`
    : health?.status === "ok"
    ? `API up - v${health.version}`
    : "Checking API...";

  return (
    <div className="flex items-center gap-2 text-xs text-[#4a5a54]">
      <span className={`inline-block w-2 h-2 rounded-full ${color}`} />
      <span>{label}</span>
    </div>
  );
}

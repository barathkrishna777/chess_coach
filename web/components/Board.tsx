"use client";

import { useEffect, useRef } from "react";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { Config } from "chessground/config";

/**
 * Slice 0 board: renders the starting position and allows free piece movement
 * on both sides. No engine, no move legality, no analysis — that arrives in
 * later slices. This component exists to prove chessground is wired up and
 * the page renders an interactive board from FEN.
 */
export default function Board() {
  const ref = useRef<HTMLDivElement>(null);
  const apiRef = useRef<Api | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    const config: Config = {
      fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
      orientation: "white",
      movable: {
        free: true,
        color: "both",
      },
      draggable: {
        enabled: true,
      },
      highlight: {
        lastMove: true,
      },
    };

    apiRef.current = Chessground(ref.current, config);

    return () => {
      apiRef.current?.destroy();
      apiRef.current = null;
    };
  }, []);

  return (
    <div
      ref={ref}
      className="w-full aspect-square rounded-md overflow-hidden shadow-2xl"
    />
  );
}

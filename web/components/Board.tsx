"use client";

import { useEffect, useRef } from "react";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { Config } from "chessground/config";
import type { Color, KeyPair } from "chessground/types";

const START_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";

type BoardProps = {
  fen: string;
  lastMove?: KeyPair | null;
  orientation?: Color;
};

export default function Board({
  fen,
  lastMove = null,
  orientation = "white",
}: BoardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const apiRef = useRef<Api | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    const config: Config = {
      fen: START_POSITION,
      orientation: "white",
      viewOnly: true,
      selectable: {
        enabled: false,
      },
      movable: {
        free: false,
      },
      draggable: {
        enabled: false,
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

  useEffect(() => {
    apiRef.current?.set({
      fen: boardFen(fen),
      orientation,
      lastMove: lastMove ? [...lastMove] : undefined,
    });
  }, [fen, lastMove, orientation]);

  return (
    <div
      ref={ref}
      className="w-full aspect-square rounded-md overflow-hidden border border-[#1f2a24]"
    />
  );
}

function boardFen(fen: string): string {
  return fen.split(" ")[0] ?? fen;
}

"use client";

import { useEffect, useRef } from "react";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { Config } from "chessground/config";
import type { Color, Dests, Key, KeyPair } from "chessground/types";

const START_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";

type BoardProps = {
  fen: string;
  lastMove?: KeyPair | null;
  orientation?: Color;
  turnColor?: Color;
  movableColor?: Color;
  legalDests?: Dests;
  disabled?: boolean;
  onMove?: (from: Key, to: Key) => void;
};

export default function Board({
  fen,
  lastMove = null,
  orientation = "white",
  turnColor = "white",
  movableColor = "white",
  legalDests,
  disabled = false,
  onMove,
}: BoardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const apiRef = useRef<Api | null>(null);
  const fenRef = useRef(fen);
  const onMoveRef = useRef<typeof onMove>(onMove);

  useEffect(() => {
    fenRef.current = fen;
    onMoveRef.current = onMove;
  }, [fen, onMove]);

  useEffect(() => {
    if (!ref.current) return;

    const config: Config = {
      fen: START_POSITION,
      orientation: "white",
      viewOnly: false,
      turnColor: "white",
      selectable: {
        enabled: false,
      },
      movable: {
        free: false,
        color: undefined,
        dests: new Map(),
        events: {
          after: (from, to) => {
            apiRef.current?.set({ fen: boardFen(fenRef.current) });
            onMoveRef.current?.(from, to);
          },
        },
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
    const canMove = Boolean(onMove) && !disabled;
    apiRef.current?.set({
      fen: boardFen(fen),
      orientation,
      turnColor,
      lastMove: lastMove ? [...lastMove] : undefined,
      viewOnly: !canMove,
      selectable: {
        enabled: canMove,
      },
      movable: {
        free: false,
        color: canMove ? movableColor : undefined,
        dests: canMove ? legalDests ?? new Map() : new Map(),
        events: {
          after: (from, to) => {
            apiRef.current?.set({ fen: boardFen(fenRef.current) });
            onMoveRef.current?.(from, to);
          },
        },
      },
      draggable: {
        enabled: canMove,
      },
    });
  }, [disabled, fen, lastMove, legalDests, movableColor, onMove, orientation, turnColor]);

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

"use client";

import { useEffect, useRef } from "react";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { Config } from "chessground/config";
import type { Color, Dests, Key, KeyPair } from "chessground/types";

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
  const initialFenRef = useRef(fen);
  const initialOrientationRef = useRef(orientation);
  const initialTurnColorRef = useRef(turnColor);
  const onMoveRef = useRef<typeof onMove>(onMove);
  const pendingLocalMoveRef = useRef(false);

  useEffect(() => {
    onMoveRef.current = onMove;
  }, [onMove]);

  useEffect(() => {
    if (!ref.current) return;

    const config: Config = {
      fen: boardFen(initialFenRef.current),
      orientation: initialOrientationRef.current,
      viewOnly: false,
      turnColor: initialTurnColorRef.current,
      selectable: {
        enabled: false,
      },
      movable: {
        free: false,
        color: undefined,
        dests: new Map(),
        events: {
          after: (from, to) => {
            pendingLocalMoveRef.current = true;
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
    const api = apiRef.current;
    if (!api) return;

    const nextFen = boardFen(fen);
    const currentFen = api.getFen();
    const shouldSyncFen = currentFen !== nextFen;
    if (pendingLocalMoveRef.current && shouldSyncFen) {
      api.state.animation.current = undefined;
    }

    api.set({
      ...(shouldSyncFen ? { fen: nextFen } : {}),
      orientation,
      turnColor,
      lastMove: lastMove ? [...lastMove] : undefined,
    });
    pendingLocalMoveRef.current = false;
  }, [fen, lastMove, orientation, turnColor]);

  useEffect(() => {
    const canMove = Boolean(onMove) && !disabled;
    apiRef.current?.set({
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
            pendingLocalMoveRef.current = true;
            onMoveRef.current?.(from, to);
          },
        },
      },
      draggable: {
        enabled: canMove,
      },
    });
  }, [disabled, legalDests, movableColor, onMove]);

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

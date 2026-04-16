"use client";

import { useEffect, useRef } from "react";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { Config } from "chessground/config";
import type { DrawShape } from "chessground/draw";
import type { Color, Dests, Key, KeyPair } from "chessground/types";

type BoardProps = {
  fen: string;
  lastMove?: KeyPair | null;
  orientation?: Color;
  turnColor?: Color;
  movableColor?: Color;
  legalDests?: Dests;
  shapes?: DrawShape[];
  disabled?: boolean;
  testId?: string;
  onMove?: (from: Key, to: Key) => void;
};

export default function Board({
  fen,
  lastMove = null,
  orientation = "white",
  turnColor = "white",
  movableColor = "white",
  legalDests,
  shapes,
  disabled = false,
  testId = "chess-board",
  onMove,
}: BoardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const apiRef = useRef<Api | null>(null);
  const initialFenRef = useRef(fen);
  const initialOrientationRef = useRef(orientation);
  const initialTurnColorRef = useRef(turnColor);
  const onMoveRef = useRef<typeof onMove>(onMove);
  const pendingLocalMoveRef = useRef<KeyPair | null>(null);
  const initialAnimationEnabledRef = useRef(!onMove);
  const hasMoveHandler = Boolean(onMove);

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
            pendingLocalMoveRef.current = [from, to];
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
      drawable: {
        enabled: false,
        visible: true,
        eraseOnClick: false,
        autoShapes: [],
      },
      animation: {
        enabled: initialAnimationEnabledRef.current,
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
    const pendingLocalMove = pendingLocalMoveRef.current;
    if (pendingLocalMove && shouldSyncFen) {
      api.state.animation.current = undefined;
      if (lastMove && !sameMove(lastMove, pendingLocalMove)) {
        api.move(lastMove[0], lastMove[1]);
      }
      if (api.getFen() !== nextFen) {
        syncFenWithoutAnimation(api, nextFen);
      }
    } else if (shouldSyncFen) {
      api.set({ fen: nextFen });
    }

    api.set({
      orientation,
      turnColor,
      lastMove: lastMove ? [...lastMove] : undefined,
    });
    pendingLocalMoveRef.current = null;
  }, [fen, lastMove, orientation, turnColor]);

  useEffect(() => {
    const canMove = hasMoveHandler && !disabled;
    apiRef.current?.set({
      animation: {
        enabled: !hasMoveHandler,
      },
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
            pendingLocalMoveRef.current = [from, to];
            onMoveRef.current?.(from, to);
          },
        },
      },
      draggable: {
        enabled: canMove,
      },
    });
  }, [disabled, hasMoveHandler, legalDests, movableColor]);

  useEffect(() => {
    apiRef.current?.set({
      drawable: {
        enabled: false,
        visible: true,
        eraseOnClick: false,
        autoShapes: shapes ?? [],
      },
    });
  }, [shapes]);

  return (
    <div
      ref={ref}
      data-testid={testId}
      className="w-full aspect-square rounded-md overflow-hidden border border-[#1f2a24]"
    />
  );
}

function boardFen(fen: string): string {
  return fen.split(" ")[0] ?? fen;
}

function syncFenWithoutAnimation(api: Api, fen: string): void {
  const wasEnabled = api.state.animation.enabled;
  api.set({ animation: { enabled: false }, fen });
  api.set({ animation: { enabled: wasEnabled } });
}

function sameMove(left: KeyPair, right: KeyPair): boolean {
  return left[0] === right[0] && left[1] === right[1];
}

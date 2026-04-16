import { Chess } from "chess.js";
import type { Dests, Key } from "chessground/types";

/**
 * Compute legal move destinations for a given FEN, formatted as a
 * chessground Dests map (from-square → to-squares[]).
 */
export function legalDestsFromFen(fen: string): Dests {
  const chess = new Chess(fen);
  const dests = new Map<Key, Key[]>();
  for (const move of chess.moves({ verbose: true })) {
    const from = move.from as Key;
    const to = move.to as Key;
    const existing = dests.get(from) ?? [];
    if (!existing.includes(to)) {
      existing.push(to);
    }
    dests.set(from, existing);
  }
  return dests;
}

/**
 * Apply a sequence of UCI moves to a starting FEN and return an array of
 * FEN strings: [startFen, afterMove1, afterMove2, ...].
 * Stops at the first move that fails to apply.
 */
export function applyUciMoves(startFen: string, ucis: string[]): string[] {
  const fens: string[] = [startFen];
  const chess = new Chess(startFen);
  for (const uci of ucis) {
    const from = uci.slice(0, 2);
    const to = uci.slice(2, 4);
    const promotion = uci.length === 5 ? uci[4] : undefined;
    try {
      chess.move({ from, to, promotion });
      fens.push(chess.fen());
    } catch {
      break;
    }
  }
  return fens;
}

/**
 * Return the side to move from a FEN string.
 */
export function sideToMoveFromFen(fen: string): "white" | "black" {
  return fen.split(" ")[1] === "b" ? "black" : "white";
}

/**
 * Parse a UCI string (e.g. "e2e4" or "e7e8q") into a [from, to] key pair.
 */
export function uciToKeyPair(uci: string): [Key, Key] {
  return [uci.slice(0, 2) as Key, uci.slice(2, 4) as Key];
}

/**
 * Return just the from+to portion of a UCI string (first 4 chars),
 * stripping any promotion piece. Useful for loose comparison.
 */
export function uciFromTo(uci: string): string {
  return uci.slice(0, 4);
}

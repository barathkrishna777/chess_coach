import type {
  AnnotatedGame,
  AnnotatedMove,
  ApiErrorEnvelope,
  EngineMove,
  MoveExplanation,
  PlayState,
} from "@/lib/types";

const API_BASE_URL = "http://localhost:8000";
const GAMES_URL = `${API_BASE_URL}/api/games`;
const EXPLAIN_URL = `${API_BASE_URL}/api/games/explain`;
const PLAY_URL = `${API_BASE_URL}/api/play`;

export async function analyzePgn(pgn: string): Promise<AnnotatedGame> {
  const response = await fetch(GAMES_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pgn }),
  });
  const body: unknown = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(body, response.status));
  }
  return body as AnnotatedGame;
}

export async function explainMove(
  move: AnnotatedMove,
  actualLine: EngineMove[],
): Promise<MoveExplanation> {
  const response = await fetch(EXPLAIN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      move,
      actual_line: actualLine,
    }),
  });
  const body: unknown = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(body, response.status));
  }
  return body as MoveExplanation;
}

export async function startPlayGame(): Promise<PlayState> {
  const response = await fetch(`${PLAY_URL}/new`, {
    method: "POST",
  });
  const body: unknown = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(body, response.status));
  }
  return body as PlayState;
}

export async function submitPlayMove(
  gameId: string,
  uci: string,
): Promise<PlayState> {
  const response = await fetch(`${PLAY_URL}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId, uci }),
  });
  const body: unknown = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(body, response.status));
  }
  return body as PlayState;
}

export async function resignPlayGame(gameId: string): Promise<PlayState> {
  const response = await fetch(`${PLAY_URL}/resign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId }),
  });
  const body: unknown = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(body, response.status));
  }
  return body as PlayState;
}

export function errorMessage(body: unknown, status: number): string {
  if (isApiErrorEnvelope(body) && body.error?.message) {
    return body.error.message;
  }
  return `Request failed with HTTP ${status}.`;
}

function isApiErrorEnvelope(value: unknown): value is ApiErrorEnvelope {
  if (!value || typeof value !== "object") return false;
  const maybeEnvelope = value as { error?: unknown };
  if (!maybeEnvelope.error || typeof maybeEnvelope.error !== "object") return false;
  return true;
}

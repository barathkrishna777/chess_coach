import type {
  AnnotatedGame,
  AnnotatedMove,
  ApiErrorEnvelope,
  EngineMove,
  ExplanationStatus,
  MoveExplanation,
  MaiaRating,
  PlayState,
  PlayOpponentRequest,
  PlayOpponentsStatus,
  ProfileDashboard,
} from "@/lib/types";

const API_BASE_URL = "http://localhost:8000";
const GAMES_URL = `${API_BASE_URL}/api/games`;
const EXPLAIN_URL = `${API_BASE_URL}/api/games/explain`;
const EXPLAIN_STATUS_URL = `${API_BASE_URL}/api/games/explain/status`;
const PLAY_URL = `${API_BASE_URL}/api/play`;
const PROFILE_URL = `${API_BASE_URL}/api/profile/me`;

export class ApiRequestError extends Error {
  readonly status: number | null;
  readonly code: string | null;
  readonly details: Record<string, string>;

  constructor({
    message,
    status,
    code,
    details = {},
  }: {
    message: string;
    status: number | null;
    code: string | null;
    details?: Record<string, string>;
  }) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export async function analyzePgn(pgn: string): Promise<AnnotatedGame> {
  return requestJson<AnnotatedGame>(GAMES_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pgn }),
  });
}

export async function explainMove(
  move: AnnotatedMove,
  actualLine: EngineMove[],
): Promise<MoveExplanation> {
  return requestJson<MoveExplanation>(EXPLAIN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      move,
      actual_line: actualLine,
    }),
  });
}

export async function getExplanationStatus(): Promise<ExplanationStatus> {
  return requestJson<ExplanationStatus>(EXPLAIN_STATUS_URL);
}

export async function startPlayGame(options?: {
  opponent: PlayOpponentRequest;
  maiaRating: MaiaRating;
}): Promise<PlayState> {
  return requestJson<PlayState>(`${PLAY_URL}/new`, {
    method: "POST",
    headers: options ? { "Content-Type": "application/json" } : undefined,
    body: options
      ? JSON.stringify({
          opponent: options.opponent,
          maia_rating: options.maiaRating,
        })
      : undefined,
  });
}

export async function getPlayOpponents(): Promise<PlayOpponentsStatus> {
  return requestJson<PlayOpponentsStatus>(`${PLAY_URL}/opponents`);
}

export async function submitPlayMove(
  gameId: string,
  uci: string,
): Promise<PlayState> {
  return requestJson<PlayState>(`${PLAY_URL}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId, uci }),
  });
}

export async function resignPlayGame(gameId: string): Promise<PlayState> {
  return requestJson<PlayState>(`${PLAY_URL}/resign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId }),
  });
}

export async function getProfileDashboard(): Promise<ProfileDashboard> {
  return requestJson<ProfileDashboard>(PROFILE_URL);
}

export function userFacingErrorMessage(error: unknown): string {
  if (error instanceof ApiRequestError) {
    if (error.status === null) {
      return "Cannot reach the local API at http://localhost:8000. Start it with make serve and retry.";
    }
    if (isPgnErrorCode(error.code)) {
      return "That PGN could not be read as a standard chess game. Check the move text and try one game at a time.";
    }
    if (error.code === "stockfish_unavailable") {
      return "Stockfish is not available. Install Stockfish or set CHESS_ML_STOCKFISH_PATH, then retry.";
    }
    if (error.code === "analysis_timeout") {
      return "Stockfish took too long to analyze this game. Try a shorter PGN or lower CHESS_ML_STOCKFISH_DEPTH.";
    }
    if (error.code === "analysis_busy") {
      return "Another review is still running. Try again in a moment.";
    }
    if (error.code === "opponent_unavailable") {
      return "No local play opponent is available. Install Stockfish or set CHESS_ML_PLAY_STOCKFISH_PATH, then retry.";
    }
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (caught: unknown) {
    throw new ApiRequestError({
      message: caught instanceof Error ? caught.message : String(caught),
      status: null,
      code: "network_error",
    });
  }

  const body = await readJson(response);
  if (!response.ok) {
    const apiError = apiErrorFromBody(body);
    throw new ApiRequestError({
      message: apiError?.message ?? `Request failed with HTTP ${response.status}.`,
      status: response.status,
      code: apiError?.code ?? null,
      details: apiError?.details ?? {},
    });
  }
  return body as T;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function isApiErrorEnvelope(value: unknown): value is ApiErrorEnvelope {
  if (!value || typeof value !== "object") return false;
  const maybeEnvelope = value as { error?: unknown };
  if (!maybeEnvelope.error || typeof maybeEnvelope.error !== "object") return false;
  return true;
}

function apiErrorFromBody(
  body: unknown,
): { code: string | null; message: string; details: Record<string, string> } | null {
  if (!isApiErrorEnvelope(body) || !body.error) return null;
  return {
    code: typeof body.error.code === "string" ? body.error.code : null,
    message:
      typeof body.error.message === "string"
        ? body.error.message
        : "The local API returned an error.",
    details: isStringRecord(body.error.details) ? body.error.details : {},
  };
}

function isStringRecord(value: unknown): value is Record<string, string> {
  if (!value || typeof value !== "object") return false;
  return Object.values(value).every((item) => typeof item === "string");
}

function isPgnErrorCode(code: string | null): boolean {
  return (
    code === "invalid_pgn" ||
    code === "multiple_games_not_supported" ||
    code === "unsupported_variant" ||
    code === "too_many_plies" ||
    code === "pgn_too_large"
  );
}

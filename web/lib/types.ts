export type Score =
  | { type: "cp"; cp: number }
  | { type: "mate"; mate_in: number; winner: "white" | "black" };

export type EngineMove = {
  uci: string;
  san: string;
};

export type EngineAnalysis = {
  status: "ok" | "terminal";
  depth: number | null;
  score: Score;
  best_move: EngineMove | null;
  pv: EngineMove[];
  nodes: number | null;
  time_ms: number;
};

export type MotifSeverity = "inaccuracy" | "mistake" | "blunder";

export type Motif = {
  id:
    | "hanging_piece"
    | "missed_tactic"
    | "allowed_tactic"
    | "endgame_slip"
    | "opening_inaccuracy";
  label: string;
  severity: MotifSeverity;
  source: "heuristic" | "learned" | "ensemble";
  score_cp: number | null;
  evidence: {
    threshold_cp: number;
    score_kind: "cp" | "mate";
    phase: "opening" | "middlegame" | "endgame";
    piece: {
      color: "white" | "black";
      role: "pawn" | "knight" | "bishop" | "rook" | "queen";
      square: string;
    } | null;
    attackers: string[];
    defenders: string[];
    best_move: EngineMove | null;
    opponent_reply: EngineMove | null;
    related_ply: number | null;
  };
};

export type MoveExplanation = {
  schema_version: "move-explanation.v1";
  status: "ok" | "unavailable" | "error";
  text: string | null;
  source: "cache" | "llm" | "fallback" | null;
  provider: "anthropic" | "codex" | "ollama" | null;
  model: string | null;
  prompt_version: "grounded-coach.v1";
  reason:
    | "api_key_missing"
    | "provider_error"
    | "invalid_response"
    | "local_model_unavailable"
    | "timeout"
    | null;
  timeout_seconds: number | null;
  retryable: boolean;
};

export type ExplanationStatus = {
  schema_version: "explanation-status.v1";
  enabled: boolean;
  configured: boolean;
  provider: "anthropic" | "codex" | "ollama" | null;
  model: string | null;
  timeout_seconds: number;
  availability: "not_checked";
  reason: "disabled" | "api_key_missing" | "unknown_provider" | null;
};

export type AnnotatedMove = {
  ply: number;
  move_number: number;
  side: "white" | "black";
  san: string;
  uci: string;
  from_square: string;
  to_square: string;
  promotion: "q" | "r" | "b" | "n" | null;
  fen_before: string;
  fen_after: string;
  analysis_before: EngineAnalysis;
  analysis_after: EngineAnalysis;
  eval_delta_cp_white: number | null;
  loss_cp: number | null;
  is_engine_best: boolean;
  motifs: Motif[];
  explanation: MoveExplanation | null;
};

export type AnnotatedGame = {
  schema_version: "annotated-game.v1";
  game_id: string;
  headers: Record<string, string>;
  players: {
    white: { name: string | null; elo: number | null };
    black: { name: string | null; elo: number | null };
  };
  result: "1-0" | "0-1" | "1/2-1/2" | "*";
  initial_fen: string;
  final_fen: string;
  analysis: {
    engine: string;
    depth: number;
    positions_evaluated: number;
    elapsed_ms: number;
  };
  moves: AnnotatedMove[];
};

export type PlayMove = {
  ply: number;
  side: "white" | "black";
  san: string;
  uci: string;
};

export type PlayOpponentRequest = "auto" | "maia" | "stockfish";
export type MaiaRating = 1100 | 1500 | 1900;

export type PlayOpponent = {
  kind: "maia" | "stockfish";
  requested: PlayOpponentRequest;
  label: string;
  engine: string;
  maia_rating: MaiaRating | null;
  fallback_reason: string | null;
};

export type PlayOpponentsStatus = {
  schema_version: "play-opponents.v1";
  default_requested: PlayOpponentRequest;
  default_maia_rating: MaiaRating;
  stockfish_path: string;
  stockfish_available: boolean;
  stockfish_label: string;
  maia: {
    lc0_path: string | null;
    lc0_available: boolean;
    weights_dir: string;
    ratings: MaiaRating[];
    available_ratings: MaiaRating[];
    missing_weights: MaiaRating[];
  };
};

export type LegalMoveDestination = {
  to_square: string;
  promotions: PromotionChoice[];
};

export type LegalMoveGroup = {
  from_square: string;
  destinations: LegalMoveDestination[];
};

export type PromotionChoice = "q" | "r" | "b" | "n";

export type PlayState = {
  schema_version: "play-state.v1";
  game_id: string;
  opponent: PlayOpponent;
  status: "active" | "completed" | "resigned";
  result: "1-0" | "0-1" | "1/2-1/2" | "*";
  fen: string;
  orientation: "white";
  legal_moves: LegalMoveGroup[];
  moves: PlayMove[];
  bot_move: EngineMove | null;
  pgn: string | null;
};

export type ProfileDashboard = {
  schema_version: "profile-dashboard.v1";
  totals: {
    games_reviewed: number;
    moves_reviewed: number;
    flagged_moves: number;
    motif_occurrences: number;
    motif_rate_per_100_moves: number;
  };
  motifs: ProfileMotifAggregate[];
  phase_breakdown: ProfilePhaseAggregate[];
  recent_games: RecentProfileGame[];
};

export type ProfileMotifAggregate = {
  id: string;
  label: string;
  count: number;
  rate_per_100_moves: number;
};

export type ProfilePhaseAggregate = {
  phase: "opening" | "middlegame" | "endgame";
  count: number;
  rate_per_100_moves: number;
};

export type RecentProfileGame = {
  game_id: string;
  players: {
    white: { name: string | null; elo: number | null };
    black: { name: string | null; elo: number | null };
  };
  result: "1-0" | "0-1" | "1/2-1/2" | "*";
  source: "pgn_upload" | "local_play";
  created_at: string;
  updated_at: string;
  ply_count: number;
  flagged_moves: number;
};

export type ApiErrorEnvelope = {
  error?: {
    code?: string;
    message?: string;
  };
};

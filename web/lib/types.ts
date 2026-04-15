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
  source: "heuristic";
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
  source: "cache" | "llm" | null;
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
  status: "active" | "completed" | "resigned";
  result: "1-0" | "0-1" | "1/2-1/2" | "*";
  fen: string;
  orientation: "white";
  legal_moves: LegalMoveGroup[];
  moves: PlayMove[];
  bot_move: EngineMove | null;
  pgn: string | null;
};

export type ApiErrorEnvelope = {
  error?: {
    code?: string;
    message?: string;
  };
};

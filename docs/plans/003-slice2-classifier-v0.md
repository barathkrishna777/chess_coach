# Plan 003 — Slice 2: Classifier v0

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-15

## Summary

Slice 2 adds a deterministic, engine-grounded motif classifier on top of the Slice 1
raw review flow. The classifier uses the existing per-move `fen_before`, `fen_after`,
`analysis_before`, `analysis_after`, `loss_cp`, and `is_engine_best` data from
`annotated-game.v1`.

No image inputs, OCR, computer vision, cloud deployment, auth, payments, or sharing are
part of this slice.

## API And Schema

Keep `schema_version: "annotated-game.v1"` and add an always-present `motifs` field to
every move as an additive contract change.

```ts
type MotifId =
  | "hanging_piece"
  | "missed_tactic"
  | "allowed_tactic"
  | "endgame_slip"
  | "opening_inaccuracy";

type MotifSeverity = "inaccuracy" | "mistake" | "blunder";
type MotifSource = "heuristic";
type GamePhase = "opening" | "middlegame" | "endgame";

type Motif = {
  id: MotifId;
  label: string;
  severity: MotifSeverity;
  source: MotifSource;
  score_cp: number | null;
  evidence: {
    threshold_cp: number;
    score_kind: "cp" | "mate";
    phase: GamePhase;
    piece: { color: "white" | "black"; role: "pawn" | "knight" | "bishop" | "rook" | "queen"; square: string } | null;
    attackers: string[];
    defenders: string[];
    best_move: { uci: string; san: string } | null;
    opponent_reply: { uci: string; san: string } | null;
    related_ply: number | null;
  };
};
```

Severity mapping:

- `50-149cp`: `inaccuracy`
- `150-299cp`: `mistake`
- `>=300cp` or mate swing: `blunder`

Existing move-level `loss_cp` stays unchanged and remains `null` for mate scores. The
classifier uses an internal effective score only to cross thresholds.

## Heuristic Algorithms

Shared helpers:

- Use python-chess boards from `fen_before` and `fen_after`.
- Compute effective mover-perspective loss from engine scores; map mate-for/against to
  a large internal value for thresholding, while returning `score_cp: null` when the raw
  score is mate-based.
- Define `is_tactical(board, move)` as legal move that captures, gives check, checkmates,
  promotes, or is the first PV move in a mate score.

Hanging piece:

- Fire when effective loss is `>=200cp`.
- In `fen_after`, scan the mover's non-pawn, non-king pieces.
- A candidate is hanging when opponent attackers outnumber friendly defenders via
  `board.attackers`.
- Attach the highest-value hanging piece, including pieces that moved onto a bad square,
  became hanging because the move uncovered them, or remained hanging after the player
  ignored the threat.

Missed tactic:

- Fire when effective loss is `>=300cp`, `analysis_before.best_move` exists, and played
  move is not the best move.
- Require the engine best move to pass `is_tactical`.
- In <=7-piece positions, emit `endgame_slip` instead of a duplicate `missed_tactic` chip
  for the same best move.

Allowed tactic:

- Fire on the current move when effective loss is `>=300cp` and the opponent's best move
  in `fen_after` passes `is_tactical`.
- This means "the move allowed an engine-grounded tactic immediately," not "the opponent
  must actually play it."
- If the next PGN move exists and matches that opponent best move, include it as
  `opponent_reply` with `related_ply`.

Endgame slip:

- Fire when `fen_before` has `<=7` total pieces including kings, effective loss is
  `>=300cp`, and the best move is tactical.
- Treat `<=7` pieces as "tablebase-relevant" eligibility without adding a Syzygy
  dependency in Slice 2.

Opening inaccuracy:

- Fire when the game starts from the standard initial FEN, `move_number <= 15`, bounded
  CP loss is `50-150cp`, and no higher-severity motif already fired.
- Do not apply this motif to PGNs that start from a custom FEN.

## Test Strategy

Use deterministic unit tests with mocked engine evaluations for most classifier
behavior. Keep Stockfish-dependent tests as smoke tests or skip when the binary is
absent.

Specific fixtures:

- Hanging piece: `1. e4 e5 2. Qh5 Nf6 3. Bc4 *` should tag `3.Bc4` as
  `hanging_piece` for the queen on `h5`.
- Missed tactic: `1. e4 d5 2. exd5 Qxd5 3. Nc3 a6 4. Nf3 *` should tag `4.Nf3` as
  `missed_tactic`, with best move `Nxd5`.
- Allowed tactic: `1. e4 d5 2. exd5 Qxd5 3. Nc3 a6 4. Nxd5 *` should tag `3...a6`
  as `allowed_tactic`, and likely also `hanging_piece`, because the queen on `d5` is
  loose.
- Endgame slip: PGN with
  `[SetUp "1"] [FEN "6k1/8/8/8/3q4/8/8/3Q2K1 w - - 0 1"] 1. Kf1 *` should tag
  `endgame_slip`, best move `Qxd4`.
- Opening inaccuracy: `1. e4 e5 2. Nf3 Nc6 3. Bc4 h6 *` with mocked ~87cp loss should
  tag `3...h6` as `opening_inaccuracy`.
- Negative controls: normal developing moves in `1.e4 e5 2.Nf3 Nc6`, custom-FEN move 1
  positions for opening exclusion, and low-loss attacked pieces below threshold.

Verification after implementation:

- `make check`
- `cd web && npm run typecheck`

## Assumptions To Validate

- Additive `motifs` under `annotated-game.v1` is acceptable.
- Mate scores should be classified via an internal effective-loss helper even though
  public `loss_cp` remains `null`.
- `allowed_tactic` means "the opponent has an immediate engine-best tactic after this
  move," whether or not the opponent actually plays it.
- `endgame_slip` replaces, rather than duplicates, `missed_tactic` for the same <=7-piece
  tactical miss.
- Hanging-piece v0 ignores pawns and uses attacker/defender counts, accepting some
  pin/SEE false positives until later refinement.

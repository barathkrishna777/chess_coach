# Plan 002 — Slice 1: Engine Wrapper + Raw Review

Status: approved for implementation
Owner: barathkrishna
Last updated: 2026-04-15

## Scope

Slice 1 turns the scaffold into a usable raw review flow:

- Parse one uploaded standard PGN into per-ply positions.
- Analyze every unique position with Stockfish.
- Return a stable annotated game JSON contract from `POST /api/games`.
- Render a local review UI with chessground, move list, eval bar, and current move details.

No motif classification, explanations, profile writes, image inputs, OCR, or computer vision are part of this slice.

## API

`POST /api/games` accepts JSON:

```json
{
  "pgn": "..."
}
```

The response is `annotated-game.v1`:

- `game_id`: deterministic `sha256:<hex>` of the normalized PGN text.
- `headers`, `players`, `result`, `initial_fen`, `final_fen`.
- `analysis`: engine name, requested depth, number of unique positions evaluated, elapsed milliseconds.
- `moves`: one entry per ply with SAN/UCI move data, FEN before/after, Stockfish analysis before/after, and centipawn loss.

Centipawn scores are reported from White's perspective. `loss_cp` is from the moving player's perspective and is `null` when mate scores prevent a bounded centipawn comparison.

## Errors

Errors use:

```json
{
  "error": {
    "code": "invalid_pgn",
    "message": "Could not parse PGN.",
    "details": {}
  }
}
```

Planned cases:

- `400 invalid_pgn`: malformed PGN, illegal moves, or zero moves.
- `400 multiple_games_not_supported`: more than one game in the upload.
- `400 unsupported_variant`: non-standard variants.
- `413 pgn_too_large`: request body exceeds the slice limit.
- `413 too_many_plies`: game exceeds the slice limit.
- `429 analysis_busy`: one full-game review is already in progress.
- `503 stockfish_unavailable`: binary missing or UCI initialization fails.
- `504 analysis_timeout`: review misses the wall-clock budget.
- `500 engine_protocol_error`: unexpected engine protocol failure.

## Stockfish Pool

The API owns a FastAPI lifespan-managed `StockfishPool`.

- Default binary: `/opt/homebrew/bin/stockfish`.
- Override: `CHESS_ML_STOCKFISH_PATH`.
- Default workers: `min(4, max(1, os.cpu_count() // 2))`.
- Override: `CHESS_ML_STOCKFISH_WORKERS`.
- Each worker runs a separate Stockfish process with `Threads=1` and modest hash.
- The review path deduplicates FENs, so an 80-ply game evaluates 81 positions, not 160.
- Slice 1 defaults to depth 16 to keep a 40-move review under the 15 second budget on local hardware; the value is configurable through `CHESS_ML_STOCKFISH_DEPTH`.
- One review request runs at a time for predictable local latency.

Cancellation and timeout behavior:

- The request has a wall-clock analysis budget of 14.5 seconds by default.
- If the request is cancelled or times out, outstanding tasks are cancelled.
- Active workers are stopped through python-chess' UCI protocol handling; unhealthy workers are discarded during pool shutdown/restart paths.
- Slice 1 returns either a complete annotated game or an error, never partial analysis.

"""Seed a local profile database with checked-in demo games."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from chess_ml.api.games import PositionEvaluator, annotate_game, profile_review_from_game
from chess_ml.engine.stockfish import (
    DEFAULT_HASH_MB,
    DEFAULT_STOCKFISH_PATH,
    StockfishPool,
    StockfishProtocolError,
    StockfishUnavailableError,
)
from chess_ml.ingestion.pgn import parse_pgn
from chess_ml.profile.store import ProfileStore

DEFAULT_DEMO_DEPTH = 6
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_PGN_PATHS: tuple[Path, ...] = (
    _REPO_ROOT / "tests" / "fixtures" / "demo" / "hanging-piece.pgn",
    _REPO_ROOT / "tests" / "fixtures" / "demo" / "missed-tactic.pgn",
    _REPO_ROOT / "tests" / "fixtures" / "demo" / "mate-threat.pgn",
)
_DEMO_REVIEWED_AT = datetime(2026, 4, 16, 12, tzinfo=UTC)


@dataclass(frozen=True)
class DemoSeedSummary:
    """Compact facts about one demo seed run."""

    db_path: Path
    games_seeded: int
    moves_reviewed: int
    motif_occurrences: int
    depth: int


async def seed_demo_profile(
    *,
    db_path: str | Path | None = None,
    pgn_paths: tuple[Path, ...] = DEMO_PGN_PATHS,
    evaluator: PositionEvaluator | None = None,
    depth: int | None = None,
) -> DemoSeedSummary:
    """Analyze checked-in demo PGNs and upsert their profile rows."""

    selected_depth = depth if depth is not None else _demo_depth_from_env()
    store = ProfileStore(db_path)

    owned_pool: StockfishPool | None = None
    selected_evaluator = evaluator
    if selected_evaluator is None:
        owned_pool = StockfishPool(
            path=os.environ.get("CHESS_ML_STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH),
            workers=_demo_workers_from_env(),
            depth=selected_depth,
            hash_mb=_demo_hash_mb_from_env(),
        )
        await owned_pool.start()
        selected_evaluator = owned_pool

    seeded_moves = 0
    seeded_motifs = 0
    try:
        for index, path in enumerate(pgn_paths):
            parsed_game = parse_pgn(path.read_text(encoding="utf-8"))
            annotated_game = await annotate_game(
                parsed_game,
                evaluator=selected_evaluator,
                depth=selected_depth,
            )
            review = profile_review_from_game(annotated_game)
            store.save_review(
                review,
                reviewed_at=_DEMO_REVIEWED_AT + timedelta(minutes=index),
            )
            seeded_moves += review.ply_count
            seeded_motifs += len(review.motif_occurrences)
    finally:
        if owned_pool is not None and owned_pool.started:
            await owned_pool.close()

    return DemoSeedSummary(
        db_path=store.path,
        games_seeded=len(pgn_paths),
        moves_reviewed=seeded_moves,
        motif_occurrences=seeded_motifs,
        depth=selected_depth,
    )


def main() -> int:
    """CLI entrypoint for `python -m chess_ml.profile.demo`."""

    try:
        summary = asyncio.run(seed_demo_profile())
    except StockfishUnavailableError as exc:
        print(
            "Stockfish is required for `make demo` because review analysis must be "
            "engine-grounded. Install Stockfish or set CHESS_ML_STOCKFISH_PATH, then retry.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1
    except StockfishProtocolError as exc:
        print(
            "Stockfish started but failed during demo analysis. Check the local binary and retry.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        "Seeded demo profile: "
        f"{summary.games_seeded} games, "
        f"{summary.moves_reviewed} reviewed moves, "
        f"{summary.motif_occurrences} motif occurrences "
        f"at depth {summary.depth} into {summary.db_path}."
    )
    return 0


def _demo_depth_from_env() -> int:
    return _env_int("CHESS_ML_DEMO_STOCKFISH_DEPTH", DEFAULT_DEMO_DEPTH)


def _demo_workers_from_env() -> int:
    return _env_int("CHESS_ML_DEMO_STOCKFISH_WORKERS", 1)


def _demo_hash_mb_from_env() -> int:
    return _env_int("CHESS_ML_DEMO_STOCKFISH_HASH_MB", DEFAULT_HASH_MB)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())

"""Slice 8 local PGN ingestion for weakly labeled classifier examples."""

from __future__ import annotations

import asyncio
import bz2
import hashlib
import io
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from chess_ml.classifier.config import LABEL_ORDER, ClassifierConfig, load_classifier_config
from chess_ml.classifier.motifs import AnalyzedMove, Motif, MotifId, detect_motifs
from chess_ml.engine.stockfish import (
    CentipawnScore,
    EngineEvaluation,
    EngineScore,
    StockfishPool,
)
from chess_ml.ingestion.pgn import ParsedPgnGame, ParsedPgnMove, PgnParseError, Side, parse_pgn


class AsyncEvaluator(Protocol):
    """Minimal async Stockfish-like evaluator used by ingestion."""

    async def evaluate(self, fen: str, *, depth: int | None = None) -> EngineEvaluation:
        """Return an engine evaluation for one FEN."""


@dataclass(frozen=True)
class LabeledPositionExample:
    """One parquet-ready weakly labeled training example."""

    game_id: str
    ply: int
    move_number: int
    side: Side
    san: str
    uci: str
    from_square: str
    to_square: str
    fen_before: str
    fen_after: str
    eval_before_cp: int | None
    eval_after_cp: int | None
    loss_cp: int | None
    is_engine_best: bool
    labels: tuple[MotifId, ...]


@dataclass(frozen=True)
class IngestionSummary:
    """A compact summary returned by the ingestion command."""

    source_pgn: Path
    dataset_path: Path
    games_read: int
    examples_written: int


async def ingest_from_config(config_path: str | Path | None = None) -> IngestionSummary:
    """Build the configured local parquet dataset using Stockfish."""

    config = (
        load_classifier_config(config_path) if config_path is not None else load_classifier_config()
    )
    pool = StockfishPool.from_env()
    await pool.start()
    try:
        return await build_dataset(config, evaluator=pool)
    finally:
        if pool.started:
            await pool.close()


async def build_dataset(
    config: ClassifierConfig,
    *,
    evaluator: AsyncEvaluator,
) -> IngestionSummary:
    """Read local PGNs, evaluate positions, weak-label them, and write parquet."""

    examples: list[LabeledPositionExample] = []
    games_read = 0
    for parsed_game in read_standard_games(
        config.source_pgn,
        max_games=config.max_games,
        max_plies=config.max_plies_per_game,
    ):
        games_read += 1
        evaluations = await evaluate_game_positions(
            parsed_game,
            evaluator=evaluator,
            depth=config.analysis_depth,
        )
        examples.extend(examples_for_game(parsed_game, evaluations))

    write_examples_parquet(examples, config.dataset_path)
    return IngestionSummary(
        source_pgn=config.source_pgn,
        dataset_path=config.dataset_path,
        games_read=games_read,
        examples_written=len(examples),
    )


def read_standard_games(
    path: str | Path,
    *,
    max_games: int,
    max_plies: int,
) -> Iterator[ParsedPgnGame]:
    """Yield standard parsed games from a local PGN-like file."""

    parsed_count = 0
    with _open_text(Path(path)) as stream:
        while parsed_count < max_games:
            game = chess.pgn.read_game(stream)
            if game is None:
                return
            try:
                parsed = parse_pgn(_game_to_pgn(game), max_plies=max_plies)
            except PgnParseError:
                continue
            parsed_count += 1
            yield parsed


async def evaluate_game_positions(
    parsed_game: ParsedPgnGame,
    *,
    evaluator: AsyncEvaluator,
    depth: int,
) -> dict[str, EngineEvaluation]:
    """Evaluate each unique position in one parsed game."""

    async def evaluate_one(fen: str) -> tuple[str, EngineEvaluation]:
        return fen, await evaluator.evaluate(fen, depth=depth)

    tasks = [asyncio.create_task(evaluate_one(fen)) for fen in _unique_positions(parsed_game)]
    results = await asyncio.gather(*tasks)
    return dict(results)


def examples_for_game(
    parsed_game: ParsedPgnGame,
    evaluations: dict[str, EngineEvaluation],
) -> list[LabeledPositionExample]:
    """Build weak-label examples for one already evaluated game."""

    analyzed_moves = [
        _analyzed_move(move, evaluations[move.fen_before], evaluations[move.fen_after])
        for move in parsed_game.moves
    ]
    motif_lists = detect_motifs(analyzed_moves, initial_fen=parsed_game.initial_fen)
    return [
        _example(
            parsed_game, move, evaluations[move.fen_before], evaluations[move.fen_after], motifs
        )
        for move, motifs in zip(parsed_game.moves, motif_lists, strict=True)
    ]


def write_examples_parquet(
    examples: Sequence[LabeledPositionExample],
    path: str | Path,
) -> None:
    """Write labeled examples to parquet."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_example_row(example) for example in examples]
    table = pa.Table.from_pylist(rows, schema=_example_schema())
    pq.write_table(table, output_path)


def read_examples_parquet(path: str | Path) -> list[dict[str, object]]:
    """Read labeled examples from parquet as plain Python rows."""

    table = pq.read_table(path)
    rows = table.to_pylist()
    return [dict(row) for row in rows]


def _example(
    parsed_game: ParsedPgnGame,
    move: ParsedPgnMove,
    analysis_before: EngineEvaluation,
    analysis_after: EngineEvaluation,
    motifs: Sequence[Motif],
) -> LabeledPositionExample:
    labels = tuple(motif.id for motif in motifs if motif.id in LABEL_ORDER)
    return LabeledPositionExample(
        game_id=_game_id(parsed_game.normalized_pgn),
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        from_square=move.from_square,
        to_square=move.to_square,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        eval_before_cp=_score_cp(analysis_before.score),
        eval_after_cp=_score_cp(analysis_after.score),
        loss_cp=_loss_cp(move.side, analysis_before.score, analysis_after.score),
        is_engine_best=(
            analysis_before.best_move is not None and analysis_before.best_move.uci == move.uci
        ),
        labels=labels,
    )


def _example_row(example: LabeledPositionExample) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": example.game_id,
        "ply": example.ply,
        "move_number": example.move_number,
        "side": example.side,
        "san": example.san,
        "uci": example.uci,
        "from_square": example.from_square,
        "to_square": example.to_square,
        "fen_before": example.fen_before,
        "fen_after": example.fen_after,
        "eval_before_cp": example.eval_before_cp,
        "eval_after_cp": example.eval_after_cp,
        "loss_cp": example.loss_cp,
        "is_engine_best": example.is_engine_best,
    }
    for label in LABEL_ORDER:
        row[f"label_{label}"] = label in example.labels
    return row


def _example_schema() -> pa.Schema:
    return pa.schema(
        [
            ("game_id", pa.string()),
            ("ply", pa.int32()),
            ("move_number", pa.int32()),
            ("side", pa.string()),
            ("san", pa.string()),
            ("uci", pa.string()),
            ("from_square", pa.string()),
            ("to_square", pa.string()),
            ("fen_before", pa.string()),
            ("fen_after", pa.string()),
            ("eval_before_cp", pa.int32()),
            ("eval_after_cp", pa.int32()),
            ("loss_cp", pa.int32()),
            ("is_engine_best", pa.bool_()),
            *(pa.field(f"label_{label}", pa.bool_()) for label in LABEL_ORDER),
        ]
    )


def _analyzed_move(
    move: ParsedPgnMove,
    analysis_before: EngineEvaluation,
    analysis_after: EngineEvaluation,
) -> AnalyzedMove:
    return AnalyzedMove(
        ply=move.ply,
        move_number=move.move_number,
        side=move.side,
        san=move.san,
        uci=move.uci,
        fen_before=move.fen_before,
        fen_after=move.fen_after,
        analysis_before=analysis_before,
        analysis_after=analysis_after,
    )


def _unique_positions(parsed_game: ParsedPgnGame) -> list[str]:
    seen: set[str] = set()
    fens: list[str] = []
    for fen in [parsed_game.initial_fen, *(move.fen_after for move in parsed_game.moves)]:
        if fen not in seen:
            seen.add(fen)
            fens.append(fen)
    return fens


def _score_cp(score: EngineScore) -> int | None:
    if isinstance(score, CentipawnScore):
        return score.cp
    return None


def _loss_cp(side: Side, before: EngineScore, after: EngineScore) -> int | None:
    if not isinstance(before, CentipawnScore) or not isinstance(after, CentipawnScore):
        return None
    delta = after.cp - before.cp
    if side == "white":
        return max(0, -delta)
    return max(0, delta)


def _game_id(normalized_pgn: str) -> str:
    return f"sha256:{hashlib.sha256(normalized_pgn.encode('utf-8')).hexdigest()}"


def _game_to_pgn(game: chess.pgn.Game) -> str:
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return str(game.accept(exporter))


@contextmanager
def _open_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8") as stream:
            yield stream
        return
    if path.suffix == ".zst":
        with path.open("rb") as raw_stream:
            decompressor = zstandard.ZstdDecompressor()
            with (
                decompressor.stream_reader(raw_stream) as reader,
                io.TextIOWrapper(reader, encoding="utf-8") as text_stream,
            ):
                yield text_stream
        return
    with path.open("r", encoding="utf-8") as stream:
        yield stream


def main() -> None:
    """CLI entrypoint for `python -m chess_ml.ingestion.lichess`."""

    summary = asyncio.run(ingest_from_config())
    print(
        "Wrote "
        f"{summary.examples_written} examples from {summary.games_read} games "
        f"to {summary.dataset_path}"
    )


if __name__ == "__main__":
    main()

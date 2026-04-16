"""Small PyTorch model and feature encoder for Slice 8 motif classification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import chess
import torch
from torch import nn

from chess_ml.classifier.config import LABEL_ORDER
from chess_ml.classifier.motifs import AnalyzedMove, MotifId
from chess_ml.engine.stockfish import CentipawnScore, EngineScore

BOARD_PLANES = 14
METADATA_FEATURES = 15
LABEL_COUNT = len(LABEL_ORDER)
CP_SCALE = 1000.0


@dataclass(frozen=True)
class EncodedMove:
    """Tensor features for one analyzed move."""

    board: torch.Tensor
    metadata: torch.Tensor


class SmallMotifNet(nn.Module):
    """A compact multi-label CNN for motif logits."""

    def __init__(
        self,
        *,
        hidden_channels: int = 32,
        dropout: float = 0.1,
        label_count: int = LABEL_COUNT,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        conv2 = hidden_channels * 2
        self.board_net = nn.Sequential(
            nn.Conv2d(BOARD_PLANES, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, conv2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv2, conv2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(conv2 + METADATA_FEATURES, conv2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conv2, label_count),
        )

    def forward(self, board: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Return one logit per motif label."""

        board_features = self.board_net(board)
        return cast(torch.Tensor, self.head(torch.cat((board_features, metadata), dim=1)))


def encode_analyzed_move(move: AnalyzedMove) -> EncodedMove:
    """Encode one runtime analyzed move for model inference."""

    return _encode(
        fen_before=move.fen_before,
        uci=move.uci,
        side=move.side,
        eval_before_cp=_score_cp(move.analysis_before.score),
        loss_cp=_loss_cp(move),
    )


def encode_row(row: Mapping[str, object]) -> EncodedMove:
    """Encode one parquet row for training or eval."""

    return _encode(
        fen_before=_str(row, "fen_before"),
        uci=_str(row, "uci"),
        side=_str(row, "side"),
        eval_before_cp=_optional_int(row.get("eval_before_cp")),
        loss_cp=_optional_int(row.get("loss_cp")),
    )


def label_vector(labels: tuple[MotifId, ...]) -> torch.Tensor:
    """Return a stable multi-hot label vector."""

    return torch.tensor(
        [1.0 if label in labels else 0.0 for label in LABEL_ORDER], dtype=torch.float32
    )


def label_vector_from_row(row: Mapping[str, object]) -> torch.Tensor:
    """Return a multi-hot label vector from parquet label columns."""

    return torch.tensor(
        [1.0 if bool(row.get(f"label_{label}", False)) else 0.0 for label in LABEL_ORDER],
        dtype=torch.float32,
    )


def labels_from_row(row: Mapping[str, object]) -> tuple[MotifId, ...]:
    """Return motif labels from parquet columns."""

    return tuple(label for label in LABEL_ORDER if bool(row.get(f"label_{label}", False)))


def stack_encoded(encoded: list[EncodedMove]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack encoded moves into batched tensors."""

    if not encoded:
        raise ValueError("Cannot stack an empty encoded move list.")
    boards = torch.stack([item.board for item in encoded])
    metadata = torch.stack([item.metadata for item in encoded])
    return boards, metadata


def _encode(
    *,
    fen_before: str,
    uci: str,
    side: str,
    eval_before_cp: int | None,
    loss_cp: int | None,
) -> EncodedMove:
    board = chess.Board(fen_before)
    move = chess.Move.from_uci(uci)
    board_tensor = torch.zeros((BOARD_PLANES, 8, 8), dtype=torch.float32)
    for square, piece in board.piece_map().items():
        plane = _piece_plane(piece)
        _set_square(board_tensor, plane, square, 1.0)

    _set_square(board_tensor, 12, move.from_square, 1.0)
    _set_square(board_tensor, 13, move.to_square, 1.0)

    metadata_values: list[float] = [
        1.0 if side == "white" else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
    ]
    ep_file = chess.square_file(board.ep_square) if board.ep_square is not None else None
    metadata_values.extend(1.0 if ep_file == file_index else 0.0 for file_index in range(8))
    metadata_values.append(_scale_cp(eval_before_cp))
    metadata_values.append(_scale_loss(loss_cp))
    return EncodedMove(
        board=board_tensor,
        metadata=torch.tensor(metadata_values, dtype=torch.float32),
    )


def _piece_plane(piece: chess.Piece) -> int:
    offset = 0 if piece.color == chess.WHITE else 6
    return offset + piece.piece_type - 1


def _set_square(tensor: torch.Tensor, plane: int, square: chess.Square, value: float) -> None:
    rank = chess.square_rank(square)
    file_index = chess.square_file(square)
    tensor[plane, rank, file_index] = value


def _score_cp(score: EngineScore) -> int | None:
    if isinstance(score, CentipawnScore):
        return score.cp
    return None


def _loss_cp(move: AnalyzedMove) -> int | None:
    before = move.analysis_before.score
    after = move.analysis_after.score
    if not isinstance(before, CentipawnScore) or not isinstance(after, CentipawnScore):
        return None
    delta = after.cp - before.cp
    if move.side == "white":
        return max(0, -delta)
    return max(0, delta)


def _scale_cp(value: int | None) -> float:
    if value is None:
        return 0.0
    return max(-1.0, min(1.0, value / CP_SCALE))


def _scale_loss(value: int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, value / CP_SCALE))


def _str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected string row value for {key}.")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values cannot be used as integer features.")
    if isinstance(value, int):
        return value
    raise ValueError(f"Expected optional integer feature, got {type(value).__name__}.")

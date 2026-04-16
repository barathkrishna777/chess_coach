"""In-memory local play sessions."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, TypeAlias, cast

import chess
import chess.pgn

from chess_ml.engine.opponent import OpponentInfo, OpponentMoveProvider

PlayStatus: TypeAlias = Literal["active", "completed", "resigned"]
GameResult: TypeAlias = Literal["1-0", "0-1", "1/2-1/2", "*"]
Side: TypeAlias = Literal["white", "black"]
Promotion: TypeAlias = Literal["q", "r", "b", "n"]


class PlayGameNotFoundError(KeyError):
    """Raised when a play session id does not exist."""


class PlayGameFinishedError(ValueError):
    """Raised when a move is submitted after a game has ended."""


class IllegalMoveError(ValueError):
    """Raised when a submitted UCI move is not legal in the current position."""


class TakebackUnavailableError(ValueError):
    """Raised when the requested takeback cannot be applied."""


class HintUnavailableError(ValueError):
    """Raised when the requested hint cannot be provided."""


@dataclass(frozen=True)
class PlayMove:
    """One move made during a play session."""

    ply: int
    side: Side
    san: str
    uci: str


@dataclass(frozen=True)
class LegalMoveDestination:
    """One destination square and optional promotion choices."""

    to_square: str
    promotions: tuple[Promotion, ...]


@dataclass(frozen=True)
class LegalMoveGroup:
    """Legal destinations grouped by origin square."""

    from_square: str
    destinations: tuple[LegalMoveDestination, ...]


@dataclass(frozen=True)
class PlayState:
    """Public play-state facts returned by the API."""

    game_id: str
    opponent: OpponentInfo
    user_color: Side
    status: PlayStatus
    result: GameResult
    fen: str
    orientation: Side
    legal_moves: tuple[LegalMoveGroup, ...]
    moves: tuple[PlayMove, ...]
    bot_move: PlayMove | None
    hints_remaining: int
    takebacks_remaining: int
    pgn: str | None


@dataclass
class PlaySession:
    """A single local game against a bot."""

    game_id: str
    opponent: OpponentInfo
    opponent_provider: OpponentMoveProvider
    user_color: Side = "white"
    board: chess.Board = field(default_factory=chess.Board)
    initial_fen: str | None = None
    moves: list[PlayMove] = field(default_factory=list)
    status: PlayStatus = "active"
    result: GameResult = "*"
    max_hints_per_game: int = 3
    hints_used: int = 0
    max_takebacks_per_game: int = 1
    takebacks_used: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.initial_fen is None:
            self.initial_fen = self.board.fen()

    async def start(self) -> PlayState:
        """Start the session, making an opening bot move when the user plays Black."""

        async with self._lock:
            self._ensure_active()
            bot_move: PlayMove | None = None
            if self.board.turn != _chess_color(self.user_color):
                working_board = self.board.copy(stack=False)
                working_moves = list(self.moves)
                bot_move = await self._apply_bot_move(working_board, working_moves)
                self.board = working_board
                self.moves = working_moves
                self.result = _result_for_board(working_board)
                self.status = "completed" if self.result != "*" else "active"
            return self.state(bot_move=bot_move)

    async def apply_user_move(self, uci: str) -> PlayState:
        """Apply a user move and, if needed, a bot reply."""

        async with self._lock:
            self._ensure_active()
            if self.board.turn != _chess_color(self.user_color):
                raise IllegalMoveError("It is not the user's turn.")

            user_move = _parse_legal_move(self.board, uci)
            working_board = self.board.copy(stack=False)
            working_moves = list(self.moves)
            working_moves.append(_play_move(working_board, user_move))
            working_board.push(user_move)

            bot_move: PlayMove | None = None
            result = _result_for_board(working_board)
            status: PlayStatus = "completed" if result != "*" else "active"

            if status == "active":
                bot_move = await self._apply_bot_move(working_board, working_moves)
                result = _result_for_board(working_board)
                status = "completed" if result != "*" else "active"

            self.board = working_board
            self.moves = working_moves
            self.status = status
            self.result = result

            return self.state(bot_move=bot_move)

    async def takeback(self) -> PlayState:
        """Undo the latest user move plus any following bot reply."""

        async with self._lock:
            self._ensure_active()
            if self.takebacks_used >= self.max_takebacks_per_game:
                raise TakebackUnavailableError(
                    "The one takeback for this game has already been used."
                )

            user_index = self._latest_user_move_index()
            if user_index is None:
                raise TakebackUnavailableError("There is no user move to take back yet.")

            remove_until = user_index + 1
            if remove_until < len(self.moves) and self.moves[remove_until].side != self.user_color:
                remove_until += 1

            remaining_moves = self.moves[:user_index] + self.moves[remove_until:]
            rebuilt_board = chess.Board(self.initial_fen or chess.STARTING_FEN)
            for played in remaining_moves:
                move = chess.Move.from_uci(played.uci)
                if move not in rebuilt_board.legal_moves:
                    raise IllegalMoveError(f"Stored move is no longer legal: {played.uci}")
                rebuilt_board.push(move)

            self.board = rebuilt_board
            self.moves = remaining_moves
            self.status = "active"
            self.result = "*"
            self.takebacks_used += 1
            return self.state(bot_move=None)

    async def hint_fen(self) -> str:
        """Return the current FEN if a hint may be requested."""

        async with self._lock:
            self._ensure_active()
            if self.board.turn != _chess_color(self.user_color):
                raise HintUnavailableError("Hints are only available when it is your turn.")
            if self.hints_used >= self.max_hints_per_game:
                raise HintUnavailableError("All three hints for this game have already been used.")
            return self.board.fen()

    async def record_hint_used(self, fen: str) -> int:
        """Consume one hint for the current position and return the remaining count."""

        async with self._lock:
            self._ensure_active()
            if self.board.fen() != fen:
                raise HintUnavailableError("The position changed before the hint could be shown.")
            if self.board.turn != _chess_color(self.user_color):
                raise HintUnavailableError("Hints are only available when it is your turn.")
            if self.hints_used >= self.max_hints_per_game:
                raise HintUnavailableError("All three hints for this game have already been used.")
            self.hints_used += 1
            return self.hints_remaining

    def resign(self) -> PlayState:
        """End the session as a user resignation."""

        self._ensure_active()
        self.status = "resigned"
        self.result = "0-1" if self.user_color == "white" else "1-0"
        return self.state(bot_move=None)

    @property
    def hints_remaining(self) -> int:
        """Number of hints still available this game."""

        return max(0, self.max_hints_per_game - self.hints_used)

    @property
    def takebacks_remaining(self) -> int:
        """Number of takebacks still available this game."""

        return max(0, self.max_takebacks_per_game - self.takebacks_used)

    def state(self, *, bot_move: PlayMove | None) -> PlayState:
        """Return the public state for this game."""

        return PlayState(
            game_id=self.game_id,
            opponent=self.opponent,
            user_color=self.user_color,
            status=self.status,
            result=self.result,
            fen=self.board.fen(),
            orientation=self.user_color,
            legal_moves=_legal_move_groups(self.board, self.status, self.user_color),
            moves=tuple(self.moves),
            bot_move=bot_move,
            hints_remaining=self.hints_remaining,
            takebacks_remaining=self.takebacks_remaining,
            pgn=self.pgn() if self.status != "active" else None,
        )

    def pgn(self) -> str:
        """Export the current game as standard PGN."""

        initial_fen = self.initial_fen or chess.STARTING_FEN
        game = chess.pgn.Game()
        if initial_fen != chess.STARTING_FEN:
            game.setup(chess.Board(initial_fen))
        game.headers["Event"] = "chess_ml local play"
        game.headers["Site"] = "localhost"
        game.headers["Date"] = datetime.now(UTC).strftime("%Y.%m.%d")
        game.headers["Round"] = "-"
        game.headers["White"] = "You" if self.user_color == "white" else self.opponent.label
        game.headers["Black"] = "You" if self.user_color == "black" else self.opponent.label
        if self.opponent.maia_rating is not None:
            elo_header = "BlackElo" if self.user_color == "white" else "WhiteElo"
            game.headers[elo_header] = str(self.opponent.maia_rating)
        game.headers["Result"] = self.result

        node: chess.pgn.GameNode = game
        board = chess.Board(initial_fen)
        for played in self.moves:
            move = chess.Move.from_uci(played.uci)
            if move not in board.legal_moves:
                raise IllegalMoveError(f"Stored move is no longer legal: {played.uci}")
            node = node.add_variation(move)
            board.push(move)

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return game.accept(exporter)

    def _ensure_active(self) -> None:
        if self.status != "active":
            raise PlayGameFinishedError("This game has already finished.")

    async def _apply_bot_move(
        self,
        working_board: chess.Board,
        working_moves: list[PlayMove],
    ) -> PlayMove:
        opponent_reply = await self.opponent_provider.choose_move(working_board.fen())
        bot_chess_move = _parse_legal_move(working_board, opponent_reply.uci)
        bot_move = _play_move(working_board, bot_chess_move)
        working_moves.append(bot_move)
        working_board.push(bot_chess_move)
        return bot_move

    def _latest_user_move_index(self) -> int | None:
        for index in range(len(self.moves) - 1, -1, -1):
            if self.moves[index].side == self.user_color:
                return index
        return None


class InMemoryPlayStore:
    """Local process memory for play sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, PlaySession] = {}

    def create(
        self,
        *,
        opponent: OpponentInfo,
        opponent_provider: OpponentMoveProvider,
        user_color: Side = "white",
    ) -> PlaySession:
        """Create and store a fresh standard game."""

        game_id = str(uuid.uuid4())
        session = PlaySession(
            game_id=game_id,
            opponent=opponent,
            opponent_provider=opponent_provider,
            user_color=user_color,
        )
        self._sessions[game_id] = session
        return session

    def get(self, game_id: str) -> PlaySession:
        """Fetch an existing game session."""

        session = self._sessions.get(game_id)
        if session is None:
            raise PlayGameNotFoundError(game_id)
        return session


def _parse_legal_move(board: chess.Board, uci: str) -> chess.Move:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as exc:
        raise IllegalMoveError(f"Invalid UCI move: {uci}") from exc
    if move not in board.legal_moves:
        raise IllegalMoveError(f"Illegal move for the current position: {uci}")
    return move


def _play_move(board: chess.Board, move: chess.Move) -> PlayMove:
    ply = board.ply() + 1
    return PlayMove(
        ply=ply,
        side="white" if board.turn == chess.WHITE else "black",
        san=board.san(move),
        uci=move.uci(),
    )


def _result_for_board(board: chess.Board) -> GameResult:
    result = board.result(claim_draw=False)
    if result in {"1-0", "0-1", "1/2-1/2", "*"}:
        return cast(GameResult, result)
    return "*"


def _legal_move_groups(
    board: chess.Board,
    status: PlayStatus,
    user_color: Side,
) -> tuple[LegalMoveGroup, ...]:
    if status != "active" or board.turn != _chess_color(user_color):
        return ()

    grouped: dict[str, dict[str, set[Promotion]]] = {}
    for move in board.legal_moves:
        from_square = chess.square_name(move.from_square)
        to_square = chess.square_name(move.to_square)
        by_destination = grouped.setdefault(from_square, {})
        promotions = by_destination.setdefault(to_square, set())
        if move.promotion is not None:
            promotion = chess.piece_symbol(move.promotion)
            if promotion in {"q", "r", "b", "n"}:
                promotions.add(cast(Promotion, promotion))

    groups: list[LegalMoveGroup] = []
    for from_square in sorted(grouped):
        destinations = tuple(
            LegalMoveDestination(
                to_square=to_square,
                promotions=tuple(sorted(promotions, key=_promotion_sort_key)),
            )
            for to_square, promotions in sorted(grouped[from_square].items())
        )
        groups.append(LegalMoveGroup(from_square=from_square, destinations=destinations))
    return tuple(groups)


def _promotion_sort_key(promotion: Promotion) -> int:
    return {"q": 0, "r": 1, "b": 2, "n": 3}[promotion]


def _chess_color(side: Side) -> bool:
    return chess.WHITE if side == "white" else chess.BLACK

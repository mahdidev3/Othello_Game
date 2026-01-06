from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.games.base import Action, GameStateProtocol
from src.games.othello import heuristics
from src.games.othello.rules import (
    PASS_ACTION,
    BOARD_SIZE,
    OthelloRules,
    bit_to_coord,
    coord_to_bit,
)
from src.utils.colors import Colorizer


@dataclass(frozen=True, slots=True)
class OthelloState(GameStateProtocol):
    """Immutable Othello/Reversi game state backed by 64-bit bitboards."""

    black: int = 0
    white: int = 0
    _player: int = OthelloRules.PLAYER_BLACK

    def __post_init__(self) -> None:
        if self.black == 0 and self.white == 0:
            b, w, player = OthelloRules.starting_position()
            object.__setattr__(self, "black", b)
            object.__setattr__(self, "white", w)
            object.__setattr__(self, "_player", player)

    @property
    def current_player(self) -> int:
        return self._player

    def legal_actions(self) -> List[Action]:
        player_bits, opp_bits = self._player_bits()
        actions = OthelloRules.legal_actions(player_bits, opp_bits, include_pass=True)
        return actions

    def apply_action(self, action: Action) -> "OthelloState":
        black, white, player = OthelloRules.apply_action(
            self.black, self.white, self._player, action
        )
        return OthelloState(black=black, white=white, _player=player)

    # Compatibility helpers
    def legal_moves(self) -> List[Action]:
        return self.legal_actions()

    def apply_move(self, move: Action) -> "OthelloState":
        return self.apply_action(move)

    def is_terminal(self) -> bool:
        return OthelloRules.is_terminal(self.black, self.white, self._player)

    def evaluate(self, player: int) -> float:
        return heuristics.evaluate_state(self, player)

    def result(self) -> Dict[str, float]:
        black_score, white_score = OthelloRules.score(self.black, self.white)
        winner = OthelloRules.winner(self.black, self.white)
        return {
            "black": float(black_score),
            "white": float(white_score),
            "winner": float(winner),
        }

    def outcome(self, perspective: int | None = None) -> float:
        winner = OthelloRules.winner(self.black, self.white)
        target = perspective if perspective is not None else self._player
        if winner == 0:
            return 0.0
        return 1.0 if winner == target else -1.0

    # Compatibility helpers for legacy code paths.
    def check_winner(self) -> int:
        return OthelloRules.winner(self.black, self.white)

    def get_force_result(self) -> int:
        return int(self.outcome(self._player))

    def get_board(self) -> List[int]:
        return OthelloRules.board_to_list(self.black, self.white)

    def _player_bits(self) -> tuple[int, int]:
        if self._player == OthelloRules.PLAYER_BLACK:
            return self.black, self.white
        return self.white, self.black

    def __str__(self) -> str:
        color = Colorizer()
        header_cols = "   " + " ".join(str(c) for c in range(BOARD_SIZE))
        rows = [header_cols]
        board = self.get_board()
        for r in range(BOARD_SIZE):
            row_cells = []
            for c in range(BOARD_SIZE):
                idx = r * BOARD_SIZE + c
                piece = board[idx]
                if piece == OthelloRules.PLAYER_BLACK:
                    symbol = color.colorize("B", fg="gray")
                elif piece == OthelloRules.PLAYER_WHITE:
                    symbol = color.colorize("W", fg="bright_white")
                else:
                    symbol = color.colorize("·", fg="bright_black")
                row_cells.append(symbol)
            rows.append(f"{r} " + " ".join(row_cells))

        legals = self.legal_actions()
        black_score, white_score = OthelloRules.score(self.black, self.white)
        turn = "BLACK" if self._player == OthelloRules.PLAYER_BLACK else "WHITE"
        info = (
            f"Turn: {turn} | Legal moves: {len(legals)} | "
            f"Score (B/W): {black_score}/{white_score}"
        )
        return "\n".join(rows + [info])

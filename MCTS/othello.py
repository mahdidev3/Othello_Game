from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .interfaces import GameState


FULL = (1 << 64) - 1
A_FILE = 0x0101010101010101
H_FILE = 0x8080808080808080
NOT_A_FILE = FULL ^ A_FILE
NOT_H_FILE = FULL ^ H_FILE


def _shift_n(x: int) -> int:
    return (x << 8) & FULL


def _shift_s(x: int) -> int:
    return x >> 8


def _shift_e(x: int) -> int:
    return ((x & NOT_H_FILE) << 1) & FULL


def _shift_w(x: int) -> int:
    return (x & NOT_A_FILE) >> 1


def _shift_ne(x: int) -> int:
    return ((x & NOT_H_FILE) << 9) & FULL


def _shift_nw(x: int) -> int:
    return ((x & NOT_A_FILE) << 7) & FULL


def _shift_se(x: int) -> int:
    return (x & NOT_H_FILE) >> 7


def _shift_sw(x: int) -> int:
    return (x & NOT_A_FILE) >> 9


_DIRS = (
    _shift_n,
    _shift_s,
    _shift_e,
    _shift_w,
    _shift_ne,
    _shift_nw,
    _shift_se,
    _shift_sw,
)


def _legal_moves(player: int, opp: int) -> int:
    empty = FULL ^ (player | opp)
    moves = 0

    for sh in _DIRS:
        t = sh(player) & opp
        t |= sh(t) & opp
        t |= sh(t) & opp
        t |= sh(t) & opp
        t |= sh(t) & opp
        t |= sh(t) & opp
        moves |= sh(t) & empty

    return moves & FULL


def _flips_for_move(move: int, player: int, opp: int) -> int:
    flips = 0
    for sh in _DIRS:
        x = sh(move)
        captured = 0
        while x and (x & opp):
            captured |= x
            x = sh(x)
        if x & player:
            flips |= captured
    return flips & FULL


def _pop_lsb(bb: int) -> Tuple[int, int]:
    lsb = bb & -bb
    idx = lsb.bit_length() - 1
    return idx, bb ^ lsb


@dataclass(frozen=True, slots=True)
class OthelloGame(GameState):
    """
    Immutable Othello/Reversi game state backed by 64-bit bitboards.

    Bit ordering matches row-major indexing:
        bit 0  -> (row=0, col=0)
        bit 63 -> (row=7, col=7)
    """

    PLAYER_1: int = 1  # black
    PLAYER_2: int = -1  # white

    black: int = 0
    white: int = 0
    _player: int = 1

    def __post_init__(self):
        if self.black == 0 and self.white == 0:
            b = (1 << (3 * 8 + 4)) | (1 << (4 * 8 + 3))
            w = (1 << (3 * 8 + 3)) | (1 << (4 * 8 + 4))
            object.__setattr__(self, "black", b)
            object.__setattr__(self, "white", w)
            object.__setattr__(self, "_player", 1)

    @property
    def current_player(self) -> int:
        return self._player

    def legal_moves(self) -> List[int]:
        player_bits, opp_bits = self._player_bits()
        moves_bb = _legal_moves(player_bits, opp_bits)
        moves: List[int] = []
        bb = moves_bb
        while bb:
            idx, bb = _pop_lsb(bb)
            moves.append(idx)
        return moves

    def apply_move(self, move: Optional[int]) -> "OthelloGame":
        if move is None:
            return OthelloGame(black=self.black, white=self.white, _player=-self._player)

        move_bit = 1 << move
        player_bits, opp_bits = self._player_bits()
        if (move_bit & _legal_moves(player_bits, opp_bits)) == 0:
            return OthelloGame(black=self.black, white=self.white, _player=-self._player)

        flips = _flips_for_move(move_bit, player_bits, opp_bits)
        player_bits |= move_bit | flips
        opp_bits ^= flips

        if self._player == self.PLAYER_1:
            black, white = player_bits, opp_bits
        else:
            black, white = opp_bits, player_bits

        return OthelloGame(black=black, white=white, _player=-self._player)

    def is_terminal(self) -> bool:
        filled = self.black | self.white
        if filled == FULL:
            return True
        player_bits, opp_bits = self._player_bits()
        if _legal_moves(player_bits, opp_bits):
            return False
        return _legal_moves(opp_bits, player_bits) == 0

    def outcome(self, perspective: Optional[int] = None) -> float:
        winner = self.check_winner()
        target = perspective if perspective is not None else self._player
        if winner == 0:
            return 0.0
        return 1.0 if winner == target else -1.0

    def check_winner(self) -> int:
        black_count = self.black.bit_count()
        white_count = self.white.bit_count()
        if black_count > white_count:
            return self.PLAYER_1
        if white_count > black_count:
            return self.PLAYER_2
        return 0

    def get_board(self) -> List[int]:
        out = [0] * 64
        b = self.black
        w = self.white
        for i in range(64):
            bit = 1 << i
            if b & bit:
                out[i] = self.PLAYER_1
            elif w & bit:
                out[i] = self.PLAYER_2
        return out

    def get_force_result(self) -> int:
        return int(self.outcome(self._player))

    def __str__(self) -> str:
        b = self.black
        w = self.white
        rows = []
        for r in range(8):
            row = []
            for c in range(8):
                idx = r * 8 + c
                bit = 1 << idx
                if b & bit:
                    row.append("●")
                elif w & bit:
                    row.append("○")
                else:
                    row.append("·")
            rows.append(" ".join(row))
        turn = "BLACK" if self._player == self.PLAYER_1 else "WHITE"
        return f"Turn: {turn}\n" + "\n".join(rows)

    def _player_bits(self) -> Tuple[int, int]:
        if self._player == self.PLAYER_1:
            return self.black, self.white
        return self.white, self.black


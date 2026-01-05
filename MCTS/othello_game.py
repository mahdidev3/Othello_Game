from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


FULL = (1 << 64) - 1
A_FILE = 0x0101010101010101
H_FILE = 0x8080808080808080
NOT_A_FILE = FULL ^ A_FILE
NOT_H_FILE = FULL ^ H_FILE


def _shift_n(x: int) -> int:
    return ((x << 8) & FULL)

def _shift_s(x: int) -> int:
    return (x >> 8)

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
    _shift_n, _shift_s, _shift_e, _shift_w,
    _shift_ne, _shift_nw, _shift_se, _shift_sw
)


def _legal_moves(player: int, opp: int) -> int:
    """
    Returns a bitboard of legal moves for `player` against `opp`.

    Standard Othello bitboard move generation:
        https://www.jaist.ac.jp/~uehara/othello/bitboard.html (conceptually)
    """
    empty = FULL ^ (player | opp)
    moves = 0

    for sh in _DIRS:
        t = sh(player) & opp
        # propagate up to 6 opponent stones (board width-2)
        t |= sh(t) & opp
        t |= sh(t) & opp
        t |= sh(t) & opp
        t |= sh(t) & opp
        t |= sh(t) & opp
        moves |= sh(t) & empty

    return moves & FULL


def _flips_for_move(move: int, player: int, opp: int) -> int:
    """
    Given a single-bit `move`, returns bitboard of opponent discs to flip.
    """
    flips = 0
    for sh in _DIRS:
        x = sh(move)
        captured = 0
        # capture contiguous opponent stones
        while x and (x & opp):
            captured |= x
            x = sh(x)
        # only valid if we end on our own stone
        if x & player:
            flips |= captured
    return flips & FULL


def _pop_lsb(bb: int) -> Tuple[int, int]:
    """
    Pops least significant 1 bit. Returns (index, new_bb).
    """
    lsb = bb & -bb
    idx = (lsb.bit_length() - 1)
    return idx, bb ^ lsb


@dataclass(frozen=True, slots=True)
class OthelloGame:
    """
    Fast immutable Othello/Reversi state using bitboards.

    Bit indexing:
      - bit 0  : (row=0, col=0)
      - bit 63 : (row=7, col=7)

    Move indexing:
      - move = row * 8 + col  in [0..63]
    """
    PLAYER_1: int = 1   # black
    PLAYER_2: int = -1  # white

    black: int = 0
    white: int = 0
    _player: int = 1

    def __post_init__(self):
        # set initial position if empty
        if self.black == 0 and self.white == 0:
            # Standard start:
            #   d4,e5 black; e4,d5 white  (with (0,0) top-left vs bottom-left ambiguity)
            # We'll use (row=3,col=3)=d4 etc with row 0 at top for printing; bit mapping uses row*8+col.
            b = (1 << (3 * 8 + 4)) | (1 << (4 * 8 + 3))
            w = (1 << (3 * 8 + 3)) | (1 << (4 * 8 + 4))
            object.__setattr__(self, "black", b)
            object.__setattr__(self, "white", w)
            object.__setattr__(self, "_player", 1)

    # ----------------------------
    # Public API expected by your code
    # ----------------------------
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

    def legal_moves(self) -> List[int]:
        p, o = self._player_bits()
        moves_bb = _legal_moves(p, o)
        moves: List[int] = []
        bb = moves_bb
        while bb:
            idx, bb = _pop_lsb(bb)
            moves.append(idx)
        return moves

    def make_move(self, move: Optional[int]) -> "OthelloGame":
        """
        Applies move and returns a new game. If move is None, perform a pass.
        """
        if move is None:
            return OthelloGame(black=self.black, white=self.white, _player=-self._player)

        move_bit = 1 << move
        p, o = self._player_bits()
        # If illegal move, treat as pass (defensive)
        if (move_bit & _legal_moves(p, o)) == 0:
            return OthelloGame(black=self.black, white=self.white, _player=-self._player)

        flips = _flips_for_move(move_bit, p, o)
        p2 = p | move_bit | flips
        o2 = o ^ flips

        if self._player == self.PLAYER_1:
            nb, nw = p2, o2
        else:
            nb, nw = o2, p2

        return OthelloGame(black=nb, white=nw, _player=-self._player)

    def is_terminal(self) -> bool:
        # terminal if neither player has legal moves OR board full
        filled = (self.black | self.white)
        if filled == FULL:
            return True
        # current moves
        p, o = self._player_bits()
        if _legal_moves(p, o):
            return False
        # opponent moves
        p2, o2 = o, p
        return _legal_moves(p2, o2) == 0

    def check_winner(self) -> int:
        b = self.black.bit_count()
        w = self.white.bit_count()
        if b > w:
            return self.PLAYER_1
        if w > b:
            return self.PLAYER_2
        return 0

    def get_force_result(self) -> int:
        """
        Returns terminal result from perspective of `self._player`:
          +1 win, -1 loss, 0 draw.
        """
        winner = self.check_winner()
        if winner == 0:
            return 0
        return 1 if winner == self._player else -1

    def __str__(self) -> str:
        # print with row 0 at top for readability
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

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _player_bits(self) -> Tuple[int, int]:
        if self._player == self.PLAYER_1:
            return self.black, self.white
        return self.white, self.black

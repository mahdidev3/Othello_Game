from __future__ import annotations

from typing import List, Optional, Tuple

from src.games.base import Action

FULL = (1 << 64) - 1
A_FILE = 0x0101010101010101
H_FILE = 0x8080808080808080
NOT_A_FILE = FULL ^ A_FILE
NOT_H_FILE = FULL ^ H_FILE

BOARD_SIZE = 8
PASS_ACTION: Action = None


def coord_to_bit(row: int, col: int) -> int:
    return 1 << (row * BOARD_SIZE + col)


def bit_to_coord(idx: int) -> Tuple[int, int]:
    row, col = divmod(idx, BOARD_SIZE)
    return row, col


def _shift_n(x: int) -> int:
    return (x << BOARD_SIZE) & FULL


def _shift_s(x: int) -> int:
    return x >> BOARD_SIZE


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


class OthelloRules:
    """Bitboard-based Othello rules helpers."""

    PLAYER_BLACK = 1
    PLAYER_WHITE = -1

    @staticmethod
    def starting_position() -> Tuple[int, int, int]:
        black = (1 << (3 * BOARD_SIZE + 4)) | (1 << (4 * BOARD_SIZE + 3))
        white = (1 << (3 * BOARD_SIZE + 3)) | (1 << (4 * BOARD_SIZE + 4))
        return black, white, OthelloRules.PLAYER_BLACK

    @staticmethod
    def legal_moves_mask(player_bits: int, opp_bits: int) -> int:
        empty = FULL ^ (player_bits | opp_bits)
        moves = 0

        for sh in _DIRS:
            t = sh(player_bits) & opp_bits
            t |= sh(t) & opp_bits
            t |= sh(t) & opp_bits
            t |= sh(t) & opp_bits
            t |= sh(t) & opp_bits
            t |= sh(t) & opp_bits
            moves |= sh(t) & empty

        return moves & FULL

    @staticmethod
    def legal_actions(player_bits: int, opp_bits: int) -> List[Action]:
        mask = OthelloRules.legal_moves_mask(player_bits, opp_bits)
        actions: List[Action] = []
        bb = mask
        while bb:
            lsb = bb & -bb
            idx = lsb.bit_length() - 1
            actions.append(bit_to_coord(idx))
            bb ^= lsb
        actions.sort()
        return actions

    @staticmethod
    def flips_for_move(move_bit: int, player_bits: int, opp_bits: int) -> int:
        flips = 0
        for sh in _DIRS:
            x = sh(move_bit)
            captured = 0
            while x and (x & opp_bits):
                captured |= x
                x = sh(x)
            if x & player_bits:
                flips |= captured
        return flips & FULL

    @staticmethod
    def apply_action(
        black: int, white: int, player: int, action: Action
    ) -> Tuple[int, int, int]:
        if action is None:
            return black, white, -player

        row, col = action
        move_bit = coord_to_bit(row, col)
        player_bits, opp_bits = (
            (black, white) if player == OthelloRules.PLAYER_BLACK else (white, black)
        )
        legal_mask = OthelloRules.legal_moves_mask(player_bits, opp_bits)
        if (move_bit & legal_mask) == 0:
            return black, white, -player

        flips = OthelloRules.flips_for_move(move_bit, player_bits, opp_bits)
        player_bits |= move_bit | flips
        opp_bits ^= flips

        if player == OthelloRules.PLAYER_BLACK:
            black, white = player_bits, opp_bits
        else:
            black, white = opp_bits, player_bits
        return black, white, -player

    @staticmethod
    def is_terminal(black: int, white: int, player: int) -> bool:
        filled = black | white
        if filled == FULL:
            return True
        player_bits, opp_bits = (
            (black, white) if player == OthelloRules.PLAYER_BLACK else (white, black)
        )
        if OthelloRules.legal_moves_mask(player_bits, opp_bits):
            return False
        return OthelloRules.legal_moves_mask(opp_bits, player_bits) == 0

    @staticmethod
    def score(black: int, white: int) -> Tuple[int, int]:
        return black.bit_count(), white.bit_count()

    @staticmethod
    def winner(black: int, white: int) -> int:
        b, w = OthelloRules.score(black, white)
        if b > w:
            return OthelloRules.PLAYER_BLACK
        if w > b:
            return OthelloRules.PLAYER_WHITE
        return 0

    @staticmethod
    def board_to_list(black: int, white: int) -> List[int]:
        board = [0] * (BOARD_SIZE * BOARD_SIZE)
        for i in range(BOARD_SIZE * BOARD_SIZE):
            bit = 1 << i
            if black & bit:
                board[i] = OthelloRules.PLAYER_BLACK
            elif white & bit:
                board[i] = OthelloRules.PLAYER_WHITE
        return board

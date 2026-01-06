from __future__ import annotations

from typing import List, Optional, Tuple

from src.games.base import GameStateProtocol
from src.games.othello.rules import BOARD_SIZE, OthelloRules

PositionalWeights = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
]


def piece_parity(state: GameStateProtocol, player: int) -> float:
    black, white = state.black, state.white  # type: ignore[attr-defined]
    b, w = OthelloRules.score(black, white)
    diff = b - w if player == OthelloRules.PLAYER_BLACK else w - b
    return diff / max(1, b + w)


def mobility_heuristic(
    state: GameStateProtocol,
    player: int,
    mask_cache: Optional[dict[tuple[int, int], int]] = None,
) -> float:
    player_bits, opp_bits = (
        (state.black, state.white)
        if player == OthelloRules.PLAYER_BLACK
        else (state.white, state.black)  # type: ignore[attr-defined]
    )
    player_moves = OthelloRules.legal_moves_mask(
        player_bits, opp_bits, cache=mask_cache
    ).bit_count()
    opp_moves = OthelloRules.legal_moves_mask(
        opp_bits, player_bits, cache=mask_cache
    ).bit_count()
    total = max(1, player_moves + opp_moves)
    return (player_moves - opp_moves) / total


def corner_heuristic(state: GameStateProtocol, player: int) -> float:
    corners = (
        0,
        BOARD_SIZE - 1,
        BOARD_SIZE * (BOARD_SIZE - 1),
        BOARD_SIZE * BOARD_SIZE - 1,
    )
    weights = 25.0
    score = 0.0
    for idx in corners:
        bit = 1 << idx
        if state.black & bit:  # type: ignore[attr-defined]
            score += weights if player == OthelloRules.PLAYER_BLACK else -weights
        elif state.white & bit:  # type: ignore[attr-defined]
            score += weights if player == OthelloRules.PLAYER_WHITE else -weights
    return score / (weights * len(corners))


def positional_heuristic(state: GameStateProtocol, player: int) -> float:
    board = OthelloRules.board_to_list(state.black, state.white)  # type: ignore[attr-defined]
    score = 0.0
    for idx, piece in enumerate(board):
        row, col = divmod(idx, BOARD_SIZE)
        weight = PositionalWeights[row][col]
        if piece == player:
            score += weight
        elif piece == -player:
            score -= weight
    return score / 100.0


def evaluate_state(
    state: GameStateProtocol,
    player: int,
    mask_cache: Optional[dict[tuple[int, int], int]] = None,
) -> float:
    """Combine multiple heuristics into a single evaluation."""

    weights = {
        "parity": 0.2,
        "mobility": 0.4,
        "corners": 0.3,
        "positional": 0.1,
    }
    return (
        weights["parity"] * piece_parity(state, player)
        + weights["mobility"] * mobility_heuristic(state, player, mask_cache=mask_cache)
        + weights["corners"] * corner_heuristic(state, player)
        + weights["positional"] * positional_heuristic(state, player)
    )

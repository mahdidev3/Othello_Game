from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.agents.base import Agent
from src.games.base import GameStateProtocol
from src.games.othello.state import OthelloState
from src.games.othello.rules import OthelloRules


@dataclass
class MatchResult:
    winner: int
    final_state: GameStateProtocol
    moves_played: int
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


def play_match(
    black: Agent,
    white: Agent,
    initial_state: Optional[GameStateProtocol] = None,
) -> MatchResult:
    """Run a single game until termination."""

    state = initial_state or OthelloState()
    black.reset()
    white.reset()

    agents = {
        OthelloRules.PLAYER_BLACK: black,
        OthelloRules.PLAYER_WHITE: white,
    }
    moves = 0

    while not state.is_terminal():
        agent = agents[state.current_player]
        action = agent.select_action(state)
        state = state.apply_action(action)
        moves += 1

    winner = OthelloRules.winner(state.black, state.white)  # type: ignore[attr-defined]
    stats = {
        black.name: black.info().as_dict(),
        white.name: white.info().as_dict(),
    }
    return MatchResult(
        winner=winner,
        final_state=state,
        moves_played=moves,
        stats=stats,
    )

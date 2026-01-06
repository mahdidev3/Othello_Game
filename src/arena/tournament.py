from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.agents.base import Agent
from src.arena.match import MatchResult, play_match
from src.games.othello.rules import OthelloRules


@dataclass
class TournamentResult:
    wins: Dict[str, int] = field(default_factory=dict)
    draws: int = 0
    games: int = 0
    timing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    nodes: Dict[str, int] = field(default_factory=dict)
    match_results: List[MatchResult] = field(default_factory=list)


def run_tournament(agent_a: Agent, agent_b: Agent, games: int = 10) -> TournamentResult:
    """Head-to-head tournament with color swapping."""

    wins = {agent_a.name: 0, agent_b.name: 0}
    draws = 0
    timing = {
        agent_a.name: {"total_time": 0.0, "moves": 0.0},
        agent_b.name: {"total_time": 0.0, "moves": 0.0},
    }
    nodes = {agent_a.name: 0, agent_b.name: 0}
    match_results: List[MatchResult] = []

    for game_idx in range(games):
        if game_idx % 2 == 0:
            black, white = agent_a, agent_b
        else:
            black, white = agent_b, agent_a

        result = play_match(black, white)
        match_results.append(result)

        if result.winner == OthelloRules.PLAYER_BLACK:
            wins[black.name] += 1
        elif result.winner == OthelloRules.PLAYER_WHITE:
            wins[white.name] += 1
        else:
            draws += 1

        for agent in (black, white):
            info = agent.info()
            timing[agent.name]["total_time"] += info.timing.total_time
            timing[agent.name]["moves"] += float(info.timing.move_count)
            nodes[agent.name] += info.nodes_expanded

    for name, data in timing.items():
        moves = data["moves"]
        data["avg_time"] = data["total_time"] / moves if moves else 0.0

    return TournamentResult(
        wins=wins,
        draws=draws,
        games=games,
        timing=timing,
        nodes=nodes,
        match_results=match_results,
    )

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List

from src.agents.base import Agent
from src.arena.tournament import TournamentResult, run_tournament


@dataclass
class BenchmarkReport:
    tournaments: List[TournamentResult] = field(default_factory=list)

    def to_json(self) -> str:
        payload = []
        for t in self.tournaments:
            payload.append(
                {
                    "wins": t.wins,
                    "draws": t.draws,
                    "games": t.games,
                    "timing": t.timing,
                    "nodes": t.nodes,
                }
            )
        return json.dumps(payload, indent=2)


def run_benchmark_suite(
    agents: List[Agent], games_per_pair: int = 4
) -> BenchmarkReport:
    tournaments: List[TournamentResult] = []
    for a, b in combinations(agents, 2):
        tournaments.append(run_tournament(a, b, games=games_per_pair))
    return BenchmarkReport(tournaments=tournaments)

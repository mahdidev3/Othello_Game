from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


class GameState(Protocol):
    """Minimal interface required by the Monte Carlo tree search implementation."""

    @property
    def current_player(self) -> int:
        ...

    def legal_moves(self) -> List[int]:
        ...

    def apply_move(self, move: Optional[int]) -> "GameState":
        ...

    def is_terminal(self) -> bool:
        ...

    def outcome(self, perspective: Optional[int] = None) -> float:
        """Return +1/-1/0 from the specified player's perspective. Defaults to the current player."""
        ...


@dataclass(slots=True)
class SearchResult:
    value: float
    move_probabilities: Dict[int, float]


@dataclass(slots=True)
class SearchNode:
    state: GameState
    parent: Optional["SearchNode"]
    prior: float = 0.0

    children: Dict[int, "SearchNode"] = field(default_factory=dict)
    untried_moves: Optional[List[int]] = None

    visits: int = 0
    value_sum: float = 0.0

    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class BaseMCTS:
    """Base class providing timing helpers and configuration for MCTS variants."""

    def __init__(self, iterations: int = 800, exploration_c: float = 1.4):
        self.iterations = int(iterations)
        self.exploration_c = float(exploration_c)
        self.search_overhead = {
            "select": {"calls": 0, "time": 0.0},
            "expand": {"calls": 0, "time": 0.0},
            "rollout": {"calls": 0, "time": 0.0},
            "backup": {"calls": 0, "time": 0.0},
        }

    def _timeit(self, key: str):
        class _Timer:
            __slots__ = ("owner", "key", "t0")

            def __init__(self, owner: "BaseMCTS", key: str):
                self.owner = owner
                self.key = key
                self.t0 = 0.0

            def __enter__(self):
                self.owner.search_overhead[self.key]["calls"] += 1
                self.t0 = time.perf_counter()

            def __exit__(self, exc_type, exc, tb):
                self.owner.search_overhead[self.key]["time"] += (
                    time.perf_counter() - self.t0
                )

        return _Timer(self, key)

    def _ucb_score(self, parent_visits: int, child: SearchNode) -> float:
        exploration = self.exploration_c * math.sqrt(
            math.log(parent_visits + 1.0) / (child.visits + 1e-9)
        )
        return child.q_value() + exploration


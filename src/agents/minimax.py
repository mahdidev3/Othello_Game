from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Callable, List, Tuple

from src.agents.base import Agent
from src.games.base import Action, GameStateProtocol
from src.utils.timing import Timer

HeuristicFn = Callable[[GameStateProtocol, int], float]


@dataclass
class MinimaxConfig:
    depth: int = 3
    heuristic: HeuristicFn | None = None


class MinimaxAgent(Agent):
    """Depth-limited minimax search."""

    def __init__(
        self,
        depth: int = 3,
        heuristic: HeuristicFn | None = None,
        seed: int | None = None,
    ):
        super().__init__(name="Minimax", seed=seed)
        self.config = MinimaxConfig(depth=depth, heuristic=heuristic)

    def select_action(self, state: GameStateProtocol) -> Action:
        perspective = state.current_player
        actions = state.legal_actions()
        with Timer() as timer:
            if not actions:
                chosen: Action = None
            else:
                best_action: Action = None
                best_value = -inf
                for action in sorted(actions):
                    value = self._min_value(
                        state.apply_action(action),
                        depth=self.config.depth - 1,
                        perspective=perspective,
                    )
                    if value > best_value:
                        best_value = value
                        best_action = action
                chosen = best_action
        self._info.timing.record(timer.elapsed)
        return chosen

    def _max_value(
        self, state: GameStateProtocol, depth: int, perspective: int
    ) -> float:
        self._info.nodes_expanded += 1
        if depth == 0 or state.is_terminal():
            return self._evaluate(state, perspective)

        value = -inf
        actions = state.legal_actions()
        if not actions:
            return self._min_value(
                state.apply_action(None), depth - 1, perspective
            )

        for action in actions:
            value = max(
                value,
                self._min_value(
                    state.apply_action(action), depth - 1, perspective
                ),
            )
        return value

    def _min_value(
        self, state: GameStateProtocol, depth: int, perspective: int
    ) -> float:
        self._info.nodes_expanded += 1
        if depth == 0 or state.is_terminal():
            return self._evaluate(state, perspective)

        value = inf
        actions = state.legal_actions()
        if not actions:
            return self._max_value(
                state.apply_action(None), depth - 1, perspective
            )

        for action in actions:
            value = min(
                value,
                self._max_value(
                    state.apply_action(action), depth - 1, perspective
                ),
            )
        return value

    def _evaluate(self, state: GameStateProtocol, player: int) -> float:
        if self.config.heuristic:
            return self.config.heuristic(state, player)
        return state.evaluate(player)

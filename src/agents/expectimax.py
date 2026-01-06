from __future__ import annotations

from math import inf
from typing import Callable

from src.agents.base import Agent
from src.games.base import Action, GameStateProtocol
from src.utils.timing import Timer

HeuristicFn = Callable[[GameStateProtocol, int], float]


class ExpectimaxAgent(Agent):
    """Expectimax search that models the opponent as a random policy."""

    def __init__(
        self,
        depth: int = 3,
        heuristic: HeuristicFn | None = None,
        seed: int | None = None,
    ):
        super().__init__(name="Expectimax", seed=seed)
        self.depth = depth
        self.heuristic = heuristic

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
                    value = self._expect_value(
                        state.apply_action(action),
                        depth=self.depth - 1,
                        maximizing=False,
                        perspective=perspective,
                    )
                    if value > best_value:
                        best_value = value
                        best_action = action
                chosen = best_action
        self._info.timing.record(timer.elapsed)
        return chosen

    def _expect_value(
        self, state: GameStateProtocol, depth: int, maximizing: bool, perspective: int
    ) -> float:
        self._info.nodes_expanded += 1
        if depth == 0 or state.is_terminal():
            return self._evaluate(state, perspective)

        actions = state.legal_actions()
        if not actions:
            return self._expect_value(
                state.apply_action(None), depth - 1, not maximizing, perspective
            )

        if maximizing:
            value = -inf
            for action in actions:
                value = max(
                    value,
                    self._expect_value(
                        state.apply_action(action), depth - 1, False, perspective
                    ),
                )
            return value

        # Opponent treated as uniform random policy.
        total = 0.0
        for action in actions:
            total += self._expect_value(
                state.apply_action(action), depth - 1, True, perspective
            )
        return total / len(actions)

    def _evaluate(self, state: GameStateProtocol, player: int) -> float:
        if self.heuristic:
            return self.heuristic(state, player)
        return state.evaluate(player)

from __future__ import annotations

from math import inf
from typing import Callable

from src.agents.base import Agent
from src.games.base import Action, GameStateProtocol
from src.utils.timing import Timer

HeuristicFn = Callable[[GameStateProtocol, int], float]


class AlphaBetaAgent(Agent):
    """Depth-limited minimax with alpha-beta pruning."""

    def __init__(
        self,
        depth: int = 4,
        heuristic: HeuristicFn | None = None,
        seed: int | None = None,
    ):
        super().__init__(name="AlphaBeta", seed=seed)
        self.depth = depth
        self.heuristic = heuristic
        self.pruned: int = 0

    def select_action(self, state: GameStateProtocol) -> Action:
        self.pruned = 0
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
                        depth=self.depth - 1,
                        alpha=-inf,
                        beta=inf,
                        perspective=perspective,
                    )
                    if value > best_value:
                        best_value = value
                        best_action = action
                chosen = best_action
        self._info.timing.record(timer.elapsed)
        self._info.extra["pruned"] = float(self.pruned)
        return chosen

    def _max_value(
        self,
        state: GameStateProtocol,
        depth: int,
        alpha: float,
        beta: float,
        perspective: int,
    ) -> float:
        self._info.nodes_expanded += 1
        if depth == 0 or state.is_terminal():
            return self._evaluate(state, perspective)

        value = -inf
        actions = state.legal_actions()
        if not actions:
            return self._min_value(
                state.apply_action(None), depth - 1, alpha, beta, perspective
            )

        for action in actions:
            value = max(
                value,
                self._min_value(
                    state.apply_action(action),
                    depth - 1,
                    alpha,
                    beta,
                    perspective,
                ),
            )
            alpha = max(alpha, value)
            if alpha >= beta:
                self.pruned += 1
                break
        return value

    def _min_value(
        self,
        state: GameStateProtocol,
        depth: int,
        alpha: float,
        beta: float,
        perspective: int,
    ) -> float:
        self._info.nodes_expanded += 1
        if depth == 0 or state.is_terminal():
            return self._evaluate(state, perspective)

        value = inf
        actions = state.legal_actions()
        if not actions:
            return self._max_value(
                state.apply_action(None), depth - 1, alpha, beta, perspective
            )

        for action in actions:
            value = min(
                value,
                self._max_value(
                    state.apply_action(action),
                    depth - 1,
                    alpha,
                    beta,
                    perspective,
                ),
            )
            beta = min(beta, value)
            if beta <= alpha:
                self.pruned += 1
                break
        return value

    def _evaluate(self, state: GameStateProtocol, player: int) -> float:
        if self.heuristic:
            return self.heuristic(state, player)
        return state.evaluate(player)

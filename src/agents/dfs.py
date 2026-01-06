from __future__ import annotations

from math import inf
from typing import Callable, List, Tuple

from src.agents.base import Agent
from src.games.base import Action, GameStateProtocol
from src.utils.timing import Timer

HeuristicFn = Callable[[GameStateProtocol, int], float]
GoalTest = Callable[[GameStateProtocol, int], bool]


class DFSAgent(Agent):
    """Depth-first search agent with horizon cut-off."""

    def __init__(
        self,
        depth_limit: int = 4,
        heuristic: HeuristicFn | None = None,
        goal_test: GoalTest | None = None,
        seed: int | None = None,
    ):
        super().__init__(name="DFS", seed=seed)
        self.depth_limit = depth_limit
        self.heuristic = heuristic
        self.goal_test = goal_test

    def select_action(self, state: GameStateProtocol) -> Action:
        perspective = state.current_player
        actions = state.legal_actions()
        with Timer() as timer:
            if not actions:
                chosen: Action = None
            else:
                best_score = -inf
                best_action: Action = actions[0]
                for action in sorted(actions):
                    score = self._dfs(
                        state.apply_action(action), 1, perspective, maximizing=False
                    )
                    if score > best_score:
                        best_score = score
                        best_action = action
                chosen = best_action
        self._info.timing.record(timer.elapsed)
        return chosen

    def _dfs(
        self,
        state: GameStateProtocol,
        depth: int,
        perspective: int,
        maximizing: bool,
    ) -> float:
        self._info.nodes_expanded += 1

        if self.goal_test and self.goal_test(state, perspective):
            return inf

        if depth >= self.depth_limit or state.is_terminal():
            return self._evaluate(state, perspective)

        actions = state.legal_actions()
        if not actions:
            return self._dfs(
                state.apply_action(None),
                depth + 1,
                perspective,
                not maximizing,
            )

        if maximizing:
            value = -inf
            for action in actions:
                value = max(
                    value,
                    self._dfs(
                        state.apply_action(action),
                        depth + 1,
                        perspective,
                        False,
                    ),
                )
            return value

        value = inf
        for action in actions:
            value = min(
                value,
                self._dfs(
                    state.apply_action(action),
                    depth + 1,
                    perspective,
                    True,
                ),
            )
        return value

    def _evaluate(self, state: GameStateProtocol, player: int) -> float:
        if self.heuristic:
            return self.heuristic(state, player)
        return state.evaluate(player)

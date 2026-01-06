from __future__ import annotations

from collections import deque
from math import inf
from typing import Callable, Deque, Tuple

from src.agents.base import Agent
from src.games.base import Action, GameStateProtocol
from src.utils.timing import Timer

HeuristicFn = Callable[[GameStateProtocol, int], float]
GoalTest = Callable[[GameStateProtocol, int], bool]


class BFSAgent(Agent):
    """Breadth-first search agent with optional goal predicate."""

    def __init__(
        self,
        depth_limit: int = 4,
        heuristic: HeuristicFn | None = None,
        goal_test: GoalTest | None = None,
        seed: int | None = None,
    ):
        super().__init__(name="BFS", seed=seed)
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
                    score = self._evaluate_action(state, action, perspective)
                    if score > best_score:
                        best_score = score
                        best_action = action
                chosen = best_action
        self._info.timing.record(timer.elapsed)
        return chosen

    def _evaluate_action(
        self, state: GameStateProtocol, action: Action, perspective: int
    ) -> float:
        queue: Deque[Tuple[GameStateProtocol, int]] = deque()
        queue.append((state.apply_action(action), 1))
        best = -inf

        while queue:
            node, depth = queue.popleft()
            self._info.nodes_expanded += 1

            if self.goal_test and self.goal_test(node, perspective):
                return inf

            if depth >= self.depth_limit or node.is_terminal():
                best = max(best, self._evaluate(node, perspective))
                continue

            legal = node.legal_actions()
            if not legal:
                queue.append((node.apply_action(None), depth + 1))
                continue
            for mv in legal:
                queue.append((node.apply_action(mv), depth + 1))
        return best

    def _evaluate(self, state: GameStateProtocol, player: int) -> float:
        if self.heuristic:
            return self.heuristic(state, player)
        return state.evaluate(player)

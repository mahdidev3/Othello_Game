from __future__ import annotations

import heapq
from typing import Callable, Dict, Tuple

from src.agents.base import Agent
from src.games.base import Action, GameStateProtocol
from src.utils.timing import Timer

HeuristicFn = Callable[[GameStateProtocol, int], float]


class AStarAgent(Agent):
    """Generic A* search agent."""

    def __init__(
        self,
        depth_limit: int = 5,
        heuristic: HeuristicFn | None = None,
        seed: int | None = None,
    ):
        super().__init__(name="AStar", seed=seed)
        self.depth_limit = depth_limit
        self.heuristic = heuristic

    def select_action(self, state: GameStateProtocol) -> Action:
        perspective = state.current_player
        actions = state.legal_actions()
        with Timer() as timer:
            if not actions:
                chosen: Action = None
            else:
                pq: list[Tuple[float, int, Action, GameStateProtocol]] = []
                visited: Dict[GameStateProtocol, float] = {}
                best_score: Dict[Action, float] = {}

                for action in actions:
                    next_state = state.apply_action(action)
                    cost = 1
                    score = self._estimate(next_state, perspective)
                    heapq.heappush(pq, (cost + score, cost, action, next_state))

                while pq:
                    _, cost, first_action, node = heapq.heappop(pq)
                    if cost > self.depth_limit:
                        continue

                    prev = visited.get(node)
                    if prev is not None and cost >= prev:
                        continue
                    visited[node] = cost
                    self._info.nodes_expanded += 1

                    if node.is_terminal() or cost == self.depth_limit:
                        value = node.evaluate(perspective)
                        best_score[first_action] = max(
                            value, best_score.get(first_action, -float("inf"))
                        )
                        continue

                    legal = node.legal_actions()
                    if not legal:
                        legal = [None]
                    for mv in legal:
                        child = node.apply_action(mv)
                        new_cost = cost + 1
                        estimate = self._estimate(child, perspective)
                        heapq.heappush(
                            pq, (new_cost + estimate, new_cost, first_action, child)
                        )

                if best_score:
                    chosen = max(
                        best_score.items(),
                        key=lambda kv: kv[1]
                        if kv[1] is not None
                        else -float("inf"),
                    )[0]
                else:
                    chosen = None
        self._info.timing.record(timer.elapsed)
        return chosen

    def _estimate(self, state: GameStateProtocol, player: int) -> float:
        heuristic_value = (
            self.heuristic(state, player) if self.heuristic else state.evaluate(player)
        )
        return -heuristic_value

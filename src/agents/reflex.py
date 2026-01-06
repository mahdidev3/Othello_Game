from __future__ import annotations

from math import inf
from typing import Callable

from src.agents.base import Agent
from src.games.base import Action, GameStateProtocol
from src.utils.timing import Timer

HeuristicFn = Callable[[GameStateProtocol, int], float]


class ReflexAgent(Agent):
    """One-ply heuristic agent."""

    def __init__(self, heuristic: HeuristicFn | None = None, seed: int | None = None):
        super().__init__(name="Reflex", seed=seed)
        self.heuristic = heuristic

    def select_action(self, state: GameStateProtocol) -> Action:
        with Timer() as timer:
            actions = state.legal_actions()
            if not actions:
                chosen: Action = None
            else:
                perspective = state.current_player
                best_score = -inf
                best_action: Action = None
                for action in sorted(actions):
                    next_state = state.apply_action(action)
                    score = (
                        self.heuristic(next_state, perspective)
                        if self.heuristic
                        else next_state.evaluate(perspective)
                    )
                    if score > best_score:
                        best_score = score
                        best_action = action
                chosen = best_action
        self._info.timing.record(timer.elapsed)
        return chosen

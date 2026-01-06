from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.agents.factory import create_agent
from src.config.config_manager import ConfigManager
from src.games.othello.rules import OthelloRules
from src.games.othello.state import OthelloState
from src.network.utils import StateConverter


class ExpertDataGenerator:
    """Generate (state, action) pairs by letting an expert agent play."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.board_size = config.get("game.board_size", 8)
        self.expert_name = config.get("expert.name", "alphabeta")
        self.expert_depth = config.get("expert.depth", 4)
        self.expert_seed = config.get("expert.seed")
        self.rollout_limit = config.get("mcts.rollout_limit", 150)
        self.iterations = config.get("mcts.iterations", 400)

        self.expert_agent = create_agent(
            self.expert_name,
            depth=self.expert_depth,
            iterations=self.iterations,
            rollout_limit=self.rollout_limit,
            seed=self.expert_seed,
        )

    def _move_to_index(self, move) -> int:
        r, c = move
        return r * self.board_size + c

    def generate_games(self, num_games: int) -> List[Tuple[np.ndarray, int]]:
        dataset: List[Tuple[np.ndarray, int]] = []
        for _ in range(num_games):
            state = OthelloState()
            # Reset any stateful metrics
            self.expert_agent.reset()
            agents = {
                OthelloRules.PLAYER_BLACK: self.expert_agent,
                OthelloRules.PLAYER_WHITE: self.expert_agent,
            }

            while not state.is_terminal():
                current_agent = agents[state.current_player]
                move = current_agent.select_action(state)

                # Skip pass-only situations
                if move is not None:
                    state_tensor = StateConverter.state_to_tensor(state)
                    action_idx = self._move_to_index(move)
                    dataset.append((state_tensor, action_idx))

                state = state.apply_action(move)

        return dataset

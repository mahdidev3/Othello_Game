from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from src.agents.factory import create_agent
from src.config.config_manager import ConfigManager
from src.games.othello.rules import OthelloRules
from src.games.othello.state import OthelloState
from src.network.utils import StateConverter
from src.utils.colors import Colorizer


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

    def generate_games(
        self, num_games: int, verbose: bool
    ) -> List[Tuple[np.ndarray, int]]:
        dataset: List[Tuple[np.ndarray, int]] = []
        color = Colorizer()

        for game_idx in range(num_games):
            state = OthelloState()
            self.expert_agent.reset()

            agents = {
                OthelloRules.PLAYER_BLACK: self.expert_agent,
                OthelloRules.PLAYER_WHITE: self.expert_agent,
            }

            moves = 0
            while not state.is_terminal():
                agent = agents[state.current_player]
                agent.clear_last_search_info()

                nodes_before = agent.info().nodes_expanded
                total_time_before = agent.info().timing.total_time

                player_label = (
                    color.colorize("BLACK", fg="bright_white", bg="black")
                    if state.current_player == OthelloRules.PLAYER_BLACK
                    else color.colorize("WHITE", fg="black", bg="bright_white")
                )

                if verbose:
                    print(
                        color.colorize(
                            f"\n=== Game {game_idx + 1}/{num_games} | "
                            f"Move {moves + 1} | {player_label} ({agent.name}) ===",
                            fg="cyan",
                        )
                    )
                    print(color.colorize("Current state:", fg="yellow"))
                    print(state)

                start_time = time.perf_counter()
                move = agent.select_action(state)
                elapsed = time.perf_counter() - start_time

                # --- dataset logging (same behavior as before) ---
                if move is not None:
                    state_tensor = StateConverter.state_to_tensor(state)
                    action_idx = self._move_to_index(move)
                    dataset.append((state_tensor, action_idx))

                # Apply move
                state = state.apply_action(move)
                moves += 1

                if verbose:
                    move_text = "PASS" if move is None else f"{move}"
                    print(color.colorize(f"Selected move: {move_text}", fg="green"))

                    move_time = (
                        elapsed
                        if elapsed >= 0
                        else agent.info().timing.total_time - total_time_before
                    )
                    print(color.colorize(f"Search time: {move_time:.6f}s", fg="yellow"))

                    nodes_after = agent.info().nodes_expanded
                    node_delta = nodes_after - nodes_before
                    if node_delta:
                        print(
                            color.colorize(
                                f"Nodes expanded this turn: {node_delta}",
                                fg="magenta",
                            )
                        )

                    extras = agent.info().extra
                    if extras:
                        print(color.colorize("Search stats:", fg="blue"))
                        for key, value in sorted(extras.items()):
                            print(color.colorize(f"  {key}: {value}", fg="blue"))

                    search_info = agent.last_search_info()
                    if search_info:
                        print(
                            color.colorize(
                                "Algorithm insights:",
                                fg="bright_white",
                                bg="black",
                            )
                        )
                        if "value" in search_info:
                            print(
                                color.colorize(
                                    f"  value: {search_info['value']:.4f}",
                                    fg="bright_white",
                                    bg="black",
                                )
                            )
                        policy = search_info.get("policy")
                        if isinstance(policy, dict) and policy:
                            top_policy = sorted(
                                policy.items(), key=lambda item: item[1], reverse=True
                            )[:5]
                            policy_str = ", ".join(
                                f"{mv}: {prob:.3f}" for mv, prob in top_policy
                            )
                            print(
                                color.colorize(
                                    f"  policy (top {len(top_policy)}): {policy_str}",
                                    fg="bright_white",
                                    bg="black",
                                )
                            )

                    print(color.colorize("Resulting state:", fg="yellow"))
                    print(state)

        return dataset

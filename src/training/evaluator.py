from typing import Any, Dict

import numpy as np
import torch

from src.agents.factory import create_agent
from src.agents.policy_net import PolicyNetAgent
from src.config.config_manager import ConfigManager
from src.games.othello.rules import OthelloRules
from src.games.othello.state import OthelloState
from src.network.utils import StateConverter


class Evaluator:
    """Evaluates a policy-only network against a baseline agent."""

    def __init__(self, model, config: ConfigManager):
        self.config = config
        self.device = config.get("device")
        self.model = model.to(self.device)
        self.model.eval()
        self.verbose = config.get("logging.verbose")
        self.board_size = getattr(model, "board_size", 8)

        # MCTS settings (used when baseline is MCTS)
        self.iterations = config.get("mcts.iterations", 400)
        self.exploration_c = config.get("mcts.exploration_c", 1.4)
        self.rollout_limit = config.get("mcts.rollout_limit", 150)

        # Evaluation settings
        self.num_eval_games = config.get("evaluation.num_games", 100)
        self.baseline_agent_name = config.get("evaluation.baseline_agent", "alphabeta")
        self.baseline_depth = config.get("evaluation.baseline_depth", 4)

    def _vprint(self, *args):
        if self.verbose:
            print(*args)

    def _print_policy_legal(self, move_probs, top_k=5):
        if not self.verbose or not move_probs:
            return

        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        for move, prob in sorted_moves:
            print(f"  move={move}  prob={prob:.4f}")

    def _make_baseline_agent(self):
        return create_agent(
            self.baseline_agent_name,
            depth=self.baseline_depth,
            iterations=self.iterations,
            rollout_limit=self.rollout_limit,
        )

    def _policy_probs(self, state: OthelloState) -> np.ndarray:
        state_tensor = StateConverter.state_to_tensor(state)
        state_tensor = (
            torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            logits = self.model(state_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model against configured baseline."""
        wins = 0
        draws = 0
        losses = 0

        policy_agent = PolicyNetAgent(self.model, device=self.device)

        for game_idx in range(self.num_eval_games):
            game = OthelloState()

            # Determine which side the model plays
            if game_idx % 2 == 0:  # Even game: model plays black (PLAYER_1)
                model_player = OthelloRules.PLAYER_BLACK
                baseline_player = OthelloRules.PLAYER_WHITE
            else:  # Odd game: model plays white (PLAYER_2)
                model_player = OthelloRules.PLAYER_WHITE
                baseline_player = OthelloRules.PLAYER_BLACK

            baseline_agent = self._make_baseline_agent()

            # Play the game
            while not game.is_terminal():
                current_player = game.current_player

                self._vprint("\n" + "=" * 40)
                self._vprint(game)  # uses __str__()
                self._vprint("=" * 40)

                if current_player == model_player:
                    self._vprint("MODEL (POLICY) TURN")

                    legals = game.legal_actions()
                    if len(legals) == 0:
                        self._vprint("No legal moves. Passing.")
                        break

                    policy_probs = self._policy_probs(game)
                    move = policy_agent.select_action(game)

                    # -------- DEBUG PRINTS --------
                    policy_legal = {}
                    for m in legals:
                        if m is None:
                            continue
                        idx = m[0] * self.board_size + m[1]
                        policy_legal[m] = policy_probs[idx]
                    top_moves = sorted(
                        policy_legal.items(), key=lambda x: x[1], reverse=True
                    )

                    self._vprint("Top policy moves:")
                    for m, p in top_moves:
                        r, c = m
                        self._vprint(f"  ({r}, {c}) : {p:.3f}")

                    r, c = move
                    self._vprint(f"Chosen move: ({r}, {c})")

                    self._vprint(
                        f"Policy sum (legal): {sum(policy_legal.values()):.3f}"
                    )

                    self._vprint("Policy (board view):")
                    for r in range(self.board_size):
                        row = []
                        for c in range(self.board_size):
                            idx = r * self.board_size + c
                            p = policy_probs[idx]
                            row.append(f"{p:5.2f}")
                        self._vprint(" ".join(row))

                else:
                    self._vprint("BASELINE TURN")

                    baseline_agent.clear_last_search_info()
                    move = baseline_agent.select_action(game)

                    policy_legal = (
                        baseline_agent.last_search_info().get("policy", {})
                        if hasattr(baseline_agent, "last_search_info")
                        else {}
                    )
                    if move is None and not policy_legal:
                        self._vprint("No legal moves available.")
                        break

                    # Show move distribution
                    if policy_legal:
                        self._print_policy_legal(policy_legal)

                    self._vprint(f"Selected move: {move}\n")

                game = game.apply_action(move)

            # Determine result
            result = OthelloRules.winner(game.black, game.white)

            if result == model_player:
                wins += 1
            elif result == baseline_player:
                losses += 1
            else:  # Draw
                draws += 1

            self._vprint("\n" + "#" * 50)
            self._vprint("FINAL BOARD")
            self._vprint(game)
            self._vprint("#" * 50)

            result = OthelloRules.winner(game.black, game.white)

            if result == model_player:
                self._vprint("RESULT: MODEL WINS ✅")
            elif result == baseline_player:
                self._vprint("RESULT: BASELINE WINS ❌")
            else:
                self._vprint("RESULT: DRAW ⚖️")

        total_games = wins + draws + losses
        win_rate = wins / total_games if total_games > 0 else 0.0

        return {"wins": wins, "draws": draws, "losses": losses, "win_rate": win_rate}

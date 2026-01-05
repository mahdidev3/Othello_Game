import torch
import numpy as np
from typing import Dict, Any
from MCTS.othello_game import OthelloGame
from MCTS.mcts_baseline import MCTSBaseline
from MCTS.mcts_model import MCTSModel
from config.config_manager import ConfigManager
from network.neural_policy_value import NeuralPolicyValue


class Evaluator:
    """Evaluates model performance against baseline MCTS."""

    def __init__(self, model, config: ConfigManager):
        self.config = config
        self.device = config.get("device")
        self.model = model.to(self.device)
        self.model.eval()
        self.verbose = config.get("logging.verbose")

        # MCTS settings
        self.iterations = config.get("mcts.iterations")
        self.exploration_c = config.get("mcts.exploration_c")

        # Evaluation settings
        self.num_eval_games = config.get("evaluation.num_games", 100)

    def _vprint(self, *args):
        if self.verbose:
            print(*args)

    def _print_policy_legal(self, move_probs, top_k=5):
        if not self.verbose:
            return

        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        for move, prob in sorted_moves:
            print(f"  move={move:2d}  prob={prob:.4f}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model against baseline MCTS."""
        wins = 0
        draws = 0
        losses = 0

        for game_idx in range(self.num_eval_games):
            game = OthelloGame()

            # Determine which side the model plays
            if game_idx % 2 == 0:  # Even game: model plays black (PLAYER_1)
                model_player = OthelloGame.PLAYER_1
                baseline_player = OthelloGame.PLAYER_2
            else:  # Odd game: model plays white (PLAYER_2)
                model_player = OthelloGame.PLAYER_2
                baseline_player = OthelloGame.PLAYER_1

            # Create MCTS instances
            nn = NeuralPolicyValue(self.model, "cuda")
            baseline_mcts = MCTSBaseline(
                iterations=self.iterations, exploration_c=self.exploration_c
            )

            # Play the game
            while not game.is_terminal():
                current_player = game._player

                self._vprint("\n" + "=" * 40)
                self._vprint(game)  # uses __str__()
                self._vprint("=" * 40)

                if current_player == model_player:
                    self._vprint("MODEL (MCTS + NN) TURN")

                    legals = game.legal_moves()
                    if len(legals) == 0:
                        self._vprint("No legal moves. Passing.")
                        break

                    move, policy_legal, policy_probs, value = nn.evaluate_state(
                        game, legals
                    )

                    # -------- DEBUG PRINTS --------
                    self._vprint(f"Value estimate: {value:+.3f}")

                    # Sort legal moves by probability
                    top_moves = sorted(
                        policy_legal.items(), key=lambda x: x[1], reverse=True
                    )[:]

                    self._vprint("Top policy moves:")
                    for m, p in top_moves:
                        r, c = divmod(m, 8)  # convert index -> board coords
                        self._vprint(f"  ({r}, {c}) : {p:.3f}")

                    r, c = divmod(move, 8)
                    self._vprint(f"Chosen move: ({r}, {c})")

                    self._vprint(
                        f"Policy sum (legal): {sum(policy_legal.values()):.3f}"
                    )

                    self._vprint("Policy (board view):")
                    for r in range(8):
                        row = []
                        for c in range(8):
                            idx = r * 8 + c
                            p = policy_probs[idx]
                            row.append(f"{p:5.2f}")
                        self._vprint(" ".join(row))

                else:
                    self._vprint("BASELINE (PURE MCTS) TURN")

                    _, policy_legal = baseline_mcts.search(game)

                    if not policy_legal:
                        self._vprint("No legal moves available.")
                        break

                    # Show move distribution
                    self._print_policy_legal(policy_legal)

                    # Deterministic selection
                    move = max(policy_legal, key=policy_legal.get)

                    self._vprint(f"Selected move: {move}\n")

                game = game.make_move(move)

            # Determine result
            result = game.check_winner()

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

            result = game.check_winner()

            if result == model_player:
                self._vprint("RESULT: MODEL WINS ✅")
            elif result == baseline_player:
                self._vprint("RESULT: BASELINE WINS ❌")
            else:
                self._vprint("RESULT: DRAW ⚖️")

        total_games = wins + draws + losses
        win_rate = wins / total_games if total_games > 0 else 0.0

        return {"wins": wins, "draws": draws, "losses": losses, "win_rate": win_rate}

import numpy as np
import torch
from MCTS.othello_game import OthelloGame
from network.utils import StateConverter
from config.config_manager import ConfigManager
from MCTS.mcts_baseline import MCTSBaseline  # Updated import


class SelfPlayGame:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.colors = config.get("logging.colors")

    def _color(self, text, code):
        return f"\033[{code}m{text}\033[0m"

    def play(self, device="cpu"):
        game = OthelloGame()
        mcts = MCTSBaseline(  # Changed to use MCTSBaseline
            iterations=self.config.get("mcts.iterations"),
            exploration_c=self.config.get("mcts.exploration_c"),
        )
        replay_data = []

        while not game.is_terminal():
            state_tensor = StateConverter.state_to_tensor(game)

            value, move_probs = mcts.search(game)

            if move_probs is None or len(move_probs.items()) == 0:
                break

            pi = np.zeros(64, dtype=np.float32)
            for move, prob in move_probs.items():
                pi[move] = prob

            replay_data.append((state_tensor, pi, None, game._player))

            if self.config.get("logging.verbose"):
                print(("#" * 100))
                print(game)
                for m, p in move_probs.items():
                    print(f"  move {m}: {p:.3f}")

            if len(move_probs.items()) == 0:
                break

            move = max(move_probs, key=move_probs.get)
            game = game.make_move(move)

        result = game.get_force_result()
        for i in range(len(replay_data)):
            state, pi, _, player = replay_data[i]
            replay_data[i] = (
                state,
                pi,
                result if player == game._player else -result,
            )

        if self.config.get("logging.verbose"):
            self._print_report(mcts, game)

        return replay_data

    def _print_report(self, mcts, game):
        print(("#" * 100))
        print(
            self._color(
                "\n=== MCTS SEARCH OVERHEAD REPORT ===\n", self.colors["header"]
            )
        )

        for name, data in mcts.search_overhead.items():
            line = (
                f"{self._color(name.ljust(15), self.colors['phase'])} | "
                f"{self._color('calls: {:8d}'.format(data['calls']), self.colors['calls'])} | "
                f"{self._color('time: {:.6f} sec'.format(data['time']), self.colors['time'])}"
            )
            print(line)

        print(("#" * 100))
        print(game)

        winner = game.check_winner()
        if winner == OthelloGame.PLAYER_1:
            print("\nWinner: PLAYER_1")
        elif winner == OthelloGame.PLAYER_2:
            print("\nWinner: PLAYER_2")
        else:
            print("\nGame ended in a draw")

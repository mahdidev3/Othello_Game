import numpy as np

from src.games.othello.state import OthelloState


class StateConverter:
    @staticmethod
    def state_to_tensor(state: OthelloState):
        board = np.array(state.get_board())
        player = state.current_player
        player_mask = (board == player).reshape(8, 8)
        opp_mask = (board == -player).reshape(8, 8)
        return np.stack([player_mask, opp_mask]).astype(np.float32)

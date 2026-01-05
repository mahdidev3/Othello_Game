import numpy as np
from MCTS.othello_game import OthelloGame


class StateConverter:
    @staticmethod
    def state_to_tensor(state: OthelloGame):
        board = np.array(state.get_board())
        player = state._player
        p = (board == player).reshape(8, 8)
        o = (board == -player).reshape(8, 8)
        return np.stack([p, o]).astype(np.float32)

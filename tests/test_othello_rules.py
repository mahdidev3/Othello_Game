from src.games.othello.rules import OthelloRules
from src.games.othello.state import OthelloState


def test_initial_legal_moves():
    state = OthelloState()
    legals = set(state.legal_actions())
    expected = {(2, 3), (3, 2), (4, 5), (5, 4)}
    assert legals == expected


def test_apply_action_flips():
    state = OthelloState()
    next_state = state.apply_action((2, 3))
    black, white = OthelloRules.score(next_state.black, next_state.white)
    assert black == 4
    assert white == 1
    assert next_state.current_player == OthelloRules.PLAYER_WHITE

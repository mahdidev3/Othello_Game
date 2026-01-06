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


def _bits_from_strings(rows):
    black = 0
    white = 0
    for r, row in enumerate(rows):
        assert len(row) == 8
        for c, ch in enumerate(row):
            bit = 1 << (r * 8 + c)
            if ch == "B":
                black |= bit
            elif ch == "W":
                white |= bit
    return black, white


def test_pass_legal_action_when_only_opponent_has_moves():
    layout = [
        "..BBBBWB",
        "BBBBBWWB",
        "BWBWWWWB",
        "BWBWWWWB",
        "BWWWBWWB",
        "BWBWWBWB",
        "BWWWWWBB",
        "BWWBBBBB",
    ]
    black, white = _bits_from_strings(layout)
    state = OthelloState(
        black=black, white=white, _player=OthelloRules.PLAYER_BLACK
    )

    assert not state.is_terminal()
    assert OthelloRules.legal_actions(black, white) == []
    assert state.legal_actions() == [None]

    passed_state = state.apply_action(None)
    assert passed_state.current_player == OthelloRules.PLAYER_WHITE
    legal_after_pass = passed_state.legal_actions()
    assert legal_after_pass
    assert all(action is not None for action in legal_after_pass)


def _bits_from_strings(rows):
    black = 0
    white = 0
    for r, row in enumerate(rows):
        assert len(row) == 8
        for c, ch in enumerate(row):
            bit = 1 << (r * 8 + c)
            if ch == "B":
                black |= bit
            elif ch == "W":
                white |= bit
    return black, white


def test_pass_legal_action_when_only_opponent_has_moves():
    layout = [
        "..BBBBWB",
        "BBBBBWWB",
        "BWBWWWWB",
        "BWBWWWWB",
        "BWWWBWWB",
        "BWBWWBWB",
        "BWWWWWBB",
        "BWWBBBBB",
    ]
    black, white = _bits_from_strings(layout)
    state = OthelloState(
        black=black, white=white, _player=OthelloRules.PLAYER_BLACK
    )

    assert not state.is_terminal()
    assert OthelloRules.legal_actions(black, white) == []
    assert state.legal_actions() == [None]

    passed_state = state.apply_action(None)
    assert passed_state.current_player == OthelloRules.PLAYER_WHITE
    legal_after_pass = passed_state.legal_actions()
    assert legal_after_pass
    assert all(action is not None for action in legal_after_pass)

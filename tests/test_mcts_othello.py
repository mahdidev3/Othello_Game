import sys
from pathlib import Path

from src.agents import MonteCarloTreeSearch  # noqa: E402
from src.games.othello.rules import OthelloRules  # noqa: E402
from src.games.othello.state import OthelloState  # noqa: E402


def _bits_from_strings(rows):
    black = 0
    white = 0
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            bit = 1 << (r * 8 + c)
            if ch == "B":
                black |= bit
            elif ch == "W":
                white |= bit
    return black, white


def test_mcts_returns_pass_when_only_pass_is_legal():
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
    state = OthelloState(black=black, white=white, _player=OthelloRules.PLAYER_BLACK)

    assert state.legal_actions() == [None]
    mcts = MonteCarloTreeSearch(iterations=10, rollout_limit=5, seed=0)

    action = mcts.select_action(state)
    assert action is None

    policy = mcts.last_search_info()["policy"]
    assert set(policy.keys()) == {None}
    # All visits should flow to the forced pass move.
    assert policy[None] == 1.0

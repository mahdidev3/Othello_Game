from src.agents import (
    AStarAgent,
    AlphaBetaAgent,
    BFSAgent,
    DFSAgent,
    ExpectimaxAgent,
    MinimaxAgent,
    MonteCarloTreeSearch,
    ReflexAgent,
)
from src.games.othello.state import OthelloState


def _is_valid_action(action):
    if action is None:
        return True
    r, c = action
    return 0 <= r < 8 and 0 <= c < 8


def test_agents_return_actions():
    state = OthelloState()
    agents = [
        ReflexAgent(),
        MinimaxAgent(depth=1),
        AlphaBetaAgent(depth=1),
        ExpectimaxAgent(depth=1),
        BFSAgent(depth_limit=1),
        DFSAgent(depth_limit=1),
        AStarAgent(depth_limit=1),
        MonteCarloTreeSearch(iterations=10, rollout_limit=5),
    ]

    for agent in agents:
        action = agent.select_action(state)
        assert _is_valid_action(action)
        info = agent.info().as_dict()
        assert "total_time" in info

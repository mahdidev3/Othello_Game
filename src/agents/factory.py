from __future__ import annotations

from typing import Callable

from src.agents.alphabeta import AlphaBetaAgent
from src.agents.astar import AStarAgent
from src.agents.bfs import BFSAgent
from src.agents.dfs import DFSAgent
from src.agents.expectimax import ExpectimaxAgent
from src.agents.mcts import MonteCarloTreeSearch
from src.agents.minimax import MinimaxAgent
from src.agents.reflex import ReflexAgent
from src.agents.base import Agent


def _create_sim_agent(name: str, seed: int | None, depth: int) -> Callable[..., Agent]:
    n = name.lower()
    if n == "reflex":
        return ReflexAgent(seed=seed)
    if n == "minimax":
        return MinimaxAgent(depth=depth, seed=seed)
    if n == "alphabeta":
        return AlphaBetaAgent(depth=depth, seed=seed)
    if n == "expectimax":
        return ExpectimaxAgent(depth=depth, seed=seed)
    if n == "bfs":
        return BFSAgent(depth_limit=depth, seed=seed)
    if n == "dfs":
        return DFSAgent(depth_limit=depth, seed=seed)
    if n == "astar":
        return AStarAgent(depth_limit=depth, seed=seed)
    raise ValueError(f"Unknown simulation agent type: {name}")


def create_agent(
    name: str,
    depth: int = 3,
    iterations: int = 400,
    rollout_limit: int = 150,
    seed: int | None = None,
    sim_agent_name: Agent | None = None,
):
    if sim_agent_name is not None:
        sim_agent = _create_sim_agent(sim_agent_name, seed, depth)
    else:
        sim_agent = None
    n = name.lower()
    if n == "reflex":
        return ReflexAgent(seed=seed)
    if n == "minimax":
        return MinimaxAgent(depth=depth, seed=seed)
    if n == "alphabeta":
        return AlphaBetaAgent(depth=depth, seed=seed)
    if n == "expectimax":
        return ExpectimaxAgent(depth=depth, seed=seed)
    if n == "bfs":
        return BFSAgent(depth_limit=depth, seed=seed)
    if n == "dfs":
        return DFSAgent(depth_limit=depth, seed=seed)
    if n == "astar":
        return AStarAgent(depth_limit=depth, seed=seed)
    if n == "mcts":
        return MonteCarloTreeSearch(
            iterations=iterations,
            rollout_limit=rollout_limit,
            seed=seed,
            sim_agent=sim_agent,
        )
    raise ValueError(f"Unknown agent type: {name}")

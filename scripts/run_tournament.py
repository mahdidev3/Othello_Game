#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.factory import create_agent  # noqa: E402
from src.arena.benchmark import run_benchmark_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-robin tournament for multiple agents."
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="List of agent names (e.g., reflex minimax alphabeta mcts).",
    )
    parser.add_argument("--games", type=int, default=2, help="Games per pairing.")
    parser.add_argument(
        "--depth", type=int, default=3, help="Search depth for tree agents."
    )
    parser.add_argument(
        "--iterations", type=int, default=400, help="Iterations for MCTS."
    )
    parser.add_argument(
        "--rollout-limit", type=int, default=150, help="Rollout cap for MCTS."
    )
    parser.add_argument(
        "--sim-agent-name",
        default="reflex",
        help="Simulation agent name for MCTS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents = [
        create_agent(
            name,
            depth=args.depth,
            iterations=args.iterations,
            rollout_limit=args.rollout_limit,
            sim_agent_name=args.sim_agent_name,
        )
        for name in args.agents
    ]
    report = run_benchmark_suite(agents, games_per_pair=args.games)
    print(report.to_json())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.factory import create_agent  # noqa: E402
from src.arena.tournament import run_tournament  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run head-to-head Othello matches.")
    parser.add_argument("--agent1", required=True, help="First agent name.")
    parser.add_argument("--agent2", required=True, help="Second agent name.")
    parser.add_argument("--games", type=int, default=2, help="Number of games to play.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for tree agents.")
    parser.add_argument("--iterations", type=int, default=400, help="Iterations for MCTS.")
    parser.add_argument("--rollout-limit", type=int, default=150, help="Rollout cap for MCTS.")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable colorful per-move logging (default: enabled).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_a = create_agent(
        args.agent1,
        depth=args.depth,
        iterations=args.iterations,
        rollout_limit=args.rollout_limit,
    )
    agent_b = create_agent(
        args.agent2,
        depth=args.depth,
        iterations=args.iterations,
        rollout_limit=args.rollout_limit,
    )

    result = run_tournament(agent_a, agent_b, games=args.games, verbose=args.verbose)
    print("=== Tournament Summary ===")
    print(f"Wins: {result.wins} | Draws: {result.draws} | Games: {result.games}")
    print("Timing (s):")
    for name, data in result.timing.items():
        print(f"  {name}: total={data['total_time']:.4f}, avg/move={data['avg_time']:.6f}")
    print("Nodes expanded:")
    for name, count in result.nodes.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()

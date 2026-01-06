#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.factory import create_agent  # noqa: E402
from src.arena.benchmark import run_benchmark_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark suite across agents.")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["reflex", "minimax", "alphabeta", "mcts"],
        help="Agents to include.",
    )
    parser.add_argument("--games", type=int, default=2, help="Games per pairing.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth.")
    parser.add_argument("--iterations", type=int, default=50, help="MCTS iterations.")
    parser.add_argument(
        "--rollout-limit", type=int, default=150, help="MCTS rollout cap."
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents = [
        create_agent(
            name,
            depth=args.depth,
            iterations=args.iterations,
            rollout_limit=args.rollout_limit,
        )
        for name in args.agents
    ]
    report = run_benchmark_suite(agents, games_per_pair=args.games)
    payload = report.to_json()

    if args.output:
        args.output.write_text(payload)
        print(f"Wrote benchmark report to {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config_manager import ConfigManager  # noqa: E402
from src.network.othello_net import OthelloNet  # noqa: E402
from src.training import (  # noqa: E402
    Evaluator,
    ExpertDataGenerator,
    ReplayBuffer,
    Trainer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior cloning training for the policy-only Othello net."
    )
    parser.add_argument(
        "--games", type=int, default=100, help="Expert games to sample."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Optimizer learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="L2 weight decay."
    )
    parser.add_argument(
        "--channels", type=int, default=128, help="Number of conv channels."
    )
    parser.add_argument("--device", default="cuda", help="Device string for torch.")
    parser.add_argument(
        "--expert-depth", type=int, default=4, help="Depth for the AlphaBeta expert."
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=10,
        help="Evaluation games versus the baseline agent.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./train_result/checkpoints",
        help="Directory for saving checkpoints.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose evaluation logging.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ConfigManager:
    cfg = ConfigManager(
        {
            "device": args.device,
            "game": {"board_size": 8},
            "expert": {"name": "alphabeta", "depth": args.expert_depth},
            "training": {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "checkpoint_dir": args.checkpoint_dir,
            },
            "logging": {"verbose": args.verbose, "colors": {}},
            "evaluation": {
                "num_games": args.eval_games,
                "baseline_agent": "alphabeta",
                "baseline_depth": args.expert_depth,
            },
            "mcts": {
                "iterations": 400,
                "exploration_c": 1.4,
                "rollout_limit": 150,
            },
        }
    )
    return cfg


def main() -> None:
    args = parse_args()
    config = build_config(args)

    device = torch.device(args.device)
    model = OthelloNet(board_size=8, channels=args.channels).to(device)
    trainer = Trainer(model, config)
    evaluator = Evaluator(model, config)
    generator = ExpertDataGenerator(config)
    replay_buffer = ReplayBuffer()

    print(
        f"Generating expert data from {args.games} games with {generator.expert_name}..."
    )
    expert_data = generator.generate_games(args.games)
    for state, action in expert_data:
        replay_buffer.add(state, action)
    print(f"Collected {len(replay_buffer)} expert samples.")

    if len(replay_buffer) == 0:
        print("No expert samples were generated; aborting training.")
        return

    steps_per_epoch = max(1, len(replay_buffer) // args.batch_size)

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            batch = replay_buffer.sample(args.batch_size)
            if not batch:
                continue
            metrics = trainer.train_step(batch)
            epoch_losses.append(metrics["total"])

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"[Epoch {epoch}] avg loss: {avg_loss:.4f}")

        eval_metrics = evaluator.evaluate()
        win_rate = eval_metrics["win_rate"]
        print(
            f"[Epoch {epoch}] Eval -> wins: {eval_metrics['wins']} draws: {eval_metrics['draws']} "
            f"losses: {eval_metrics['losses']} win_rate: {win_rate:.3f}"
        )

        if epoch % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"epoch_{epoch}.pth"
            trainer.save_checkpoint(str(ckpt_path), win_rate=win_rate)


if __name__ == "__main__":
    main()

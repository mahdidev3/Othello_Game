import os
import torch
import torch.nn as nn
import torch.optim as optim
from config.config_manager import ConfigManager


class Trainer:
    def __init__(self, model, config: ConfigManager):
        self.config = config
        self.device = config.get("device")
        self.model = model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("training.learning_rate"),
            weight_decay=config.get("training.weight_decay"),
        )

        self.value_loss_fn = nn.MSELoss()
        self.step = 0
        self.best_win_rate = 0.0

        # Checkpoint settings
        self.checkpoint_frequency = config.get("training.checkpoint_frequency", 1000)
        self.checkpoint_dir = config.get("training.checkpoint_dir", "checkpoints")

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_step(self, batch):
        states, policies, values = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        policies = torch.tensor(policies, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)

        pred_policy, pred_value = self.model(states)

        policy_loss = -torch.mean(torch.sum(policies * pred_policy, dim=1))
        value_loss = self.value_loss_fn(pred_value.squeeze(-1), values)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.step += 1

        # Save checkpoint at specified frequency
        if self.step % self.checkpoint_frequency == 0:
            self.save_checkpoint(
                f"{self.checkpoint_dir}/checkpoint_{self.step}.pth", win_rate=0.0
            )
            self.save_checkpoint(
                f"{self.checkpoint_dir}/last_checkpoint.pth", win_rate=0.0
            )

        return {
            "total": total_loss.item(),
            "policy": policy_loss.item(),
            "value": value_loss.item(),
        }

    def save_checkpoint(self, filename, win_rate=0.0):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "win_rate": win_rate,
                "best_win_rate": self.best_win_rate,
            },
            filename,
        )
        print(
            f"Checkpoint saved: {filename} at step {self.step}, win_rate: {win_rate:.4f}"
        )

        # Update best checkpoint if win rate is better
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            torch.save(
                {
                    "step": self.step,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "win_rate": win_rate,
                    "best_win_rate": self.best_win_rate,
                },
                f"{self.checkpoint_dir}/best_checkpoint.pth",
            )
            print(f"Best checkpoint updated: win_rate: {win_rate:.4f}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.step = checkpoint.get("step", 0)
        self.best_win_rate = checkpoint.get("best_win_rate", 0.0)
        print(
            f"Checkpoint loaded: {filename} at step {self.step}, best_win_rate: {self.best_win_rate:.4f}"
        )

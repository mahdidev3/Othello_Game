from __future__ import annotations

import numpy as np
import torch

from src.agents.base import Agent
from src.games.base import Action, GameStateProtocol
from src.network.utils import StateConverter


class PolicyNetAgent(Agent):
    """Greedy agent powered by a policy-only neural network."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__(name="PolicyNet")
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.board_size = getattr(model, "board_size", 8)

    def select_action(self, state: GameStateProtocol) -> Action:
        legal_moves = state.legal_actions()
        if not legal_moves:
            return None

        state_tensor = StateConverter.state_to_tensor(state)
        state_tensor = (
            torch.tensor(state_tensor, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(state_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        mask = np.zeros_like(probs, dtype=np.float32)
        for move in legal_moves:
            if move is None:
                continue
            r, c = move
            mask[r * self.board_size + c] = 1.0

        probs *= mask
        if probs.sum() == 0:
            return None

        best_idx = int(np.argmax(probs))
        row, col = divmod(best_idx, self.board_size)
        return (row, col)

import numpy as np
import torch

from src.games.othello.state import OthelloState
from src.network.utils import StateConverter


class NeuralPolicyValue:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_state(self, state: OthelloState, legal_moves):
        """
        Args:
            state: game state object
            legal_moves: list of legal move coordinate tuples or None for pass

        Returns:
            move: selected move (int)
            policy_legal: dict {move: prob}
            policy_full: np.ndarray shape (64,)
        """

        # 1. Convert state -> tensor
        state_tensor = StateConverter.state_to_tensor(state)
        state_tensor = (
            torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # 2. Forward pass
        with torch.no_grad():
            policy_logits = self.model(state_tensor)

        # 3. Policy processing
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        board_size = int(len(policy_probs) ** 0.5)

        # 4. Mask illegal moves (expecting (row, col) tuples)
        mask = np.zeros_like(policy_probs, dtype=np.float32)
        for move in legal_moves:
            if move is None:
                continue
            row, col = move
            idx = row * board_size + col
            mask[idx] = 1.0

        policy_probs *= mask

        # Handle all-zero (no legal moves edge case)
        if policy_probs.sum() > 0:
            policy_probs /= policy_probs.sum()
        else:
            # Pass move case
            policy_probs = mask / mask.sum()

        # 5. Extract legal policy
        policy_legal = {}
        for move in legal_moves:
            if move is None:
                continue
            row, col = move
            idx = row * board_size + col
            policy_legal[move] = policy_probs[idx]

        # 6. Choose move (deterministic for eval)
        move = max(policy_legal, key=policy_legal.get)

        return move, policy_legal, policy_probs

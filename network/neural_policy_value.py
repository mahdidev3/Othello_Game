import numpy as np
import torch
from network.utils import StateConverter


class NeuralPolicyValue:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_state(self, state, legal_moves):
        """
        Args:
            state: game state object
            legal_moves: list of legal move indices (0..63)

        Returns:
            move: selected move (int)
            policy_legal: dict {move: prob}
            policy_full: np.ndarray shape (64,)
            value: float in [-1, 1]
        """

        # 1. Convert state -> tensor
        state_tensor = StateConverter.state_to_tensor(state)
        state_tensor = (
            torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # 2. Forward pass
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)

        # 3. Policy processing
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.item()

        # 4. Mask illegal moves
        mask = np.zeros_like(policy_probs, dtype=np.float32)
        mask[legal_moves] = 1.0

        policy_probs *= mask

        # Handle all-zero (no legal moves edge case)
        if policy_probs.sum() > 0:
            policy_probs /= policy_probs.sum()
        else:
            # Pass move case
            policy_probs = mask / mask.sum()

        # 5. Extract legal policy
        policy_legal = {move: policy_probs[move] for move in legal_moves}

        # 6. Choose move (deterministic for eval)
        move = max(policy_legal, key=policy_legal.get)

        return move, policy_legal, policy_probs, value

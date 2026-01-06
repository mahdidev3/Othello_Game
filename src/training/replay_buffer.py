import pickle
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(list(self.buffer), f)
        print(f"ReplayBuffer saved to {filename} (size={len(self.buffer)})")

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.buffer = deque(data, maxlen=self.buffer.maxlen)
        print(f"ReplayBuffer loaded from {filename} (size={len(self.buffer)})")

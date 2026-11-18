"""DQN replay buffer. Policy gradient algorithms do not need this module."""
from copy import deepcopy
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        self.capacity = capacity
        self.data = []
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def push(self, transition):
        # Store snapshots so later board mutations cannot alter past transitions.
        item = deepcopy(transition)
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = self.rng.choice(len(self.data), batch_size, replace=False)
        return [self.data[int(i)] for i in indices]

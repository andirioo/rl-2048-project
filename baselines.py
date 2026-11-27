"""Two baselines requiring no training, both using the shared evaluator."""
import numpy as np
from env import move_board
from agents.base_agent import ActionResult, BaseAgent


class BaselineAgent(BaseAgent):
    def __init__(self, strategy="random", seed=0):
        if strategy not in {"random", "greedy"}:
            raise ValueError("baseline must be random or greedy")
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def act(self, state, action_mask, training=False):
        legal = np.flatnonzero(action_mask)
        if not len(legal):
            raise ValueError("No legal actions are available")
        if self.strategy == "random":
            action = int(self.rng.choice(legal))
        else:
            # Compare immediate merge scores; break ties uniformly at random.
            scores = np.array([move_board(state, int(a))[1] for a in legal])
            action = int(self.rng.choice(legal[scores == scores.max()]))
        return ActionResult(action)

    def observe(self, transition):
        pass

    def learn(self):
        return {}

    def save(self, path):
        raise NotImplementedError("Baselines do not require checkpoints; use --baseline")

    def load(self, path):
        raise NotImplementedError("Baselines do not require checkpoints; use --baseline")

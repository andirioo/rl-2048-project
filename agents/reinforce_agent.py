"""Alternative algorithm: Monte Carlo REINFORCE, without a Q-network or replay buffer.

Update from complete, naturally terminated episodes. Discard budget-truncated trajectories.
"""
import numpy as np
import torch
from torch.distributions import Categorical
from agents.base_agent import ActionResult, TorchAgent


def discounted_returns(rewards, gamma):
    result = np.zeros(len(rewards), dtype=np.float32)
    future = 0.0
    for i in reversed(range(len(rewards))):
        future = float(rewards[i]) + gamma * future
        result[i] = future
    return result


class ReinforceAgent(TorchAgent):
    def __init__(self, config):
        super().__init__(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.trajectory = []
        self.ready = False

    @torch.no_grad()
    def act(self, state, action_mask, training=False):
        mask = self.check_mask(action_mask)
        self.model.eval()
        logits = self.model(self.encode_batch([state]))[0]
        logits = logits.masked_fill(~torch.as_tensor(mask, device=self.device), -torch.inf)
        distribution = Categorical(logits=logits)
        if training:
            action = int(self.rng.choice(4, p=distribution.probs.cpu().numpy().astype(float)
                                        / distribution.probs.cpu().numpy().sum(dtype=float)))
        else:
            action = int(logits.argmax().item())
        log_prob = float(distribution.log_prob(torch.tensor(action, device=self.device)).item())
        return ActionResult(action, {"log_prob": log_prob})

    def observe(self, transition):
        if self.ready:
            raise RuntimeError("Call learn() after collecting a complete episode")
        self.env_steps += 1
        self.trajectory.append(transition)
        if transition.terminated:
            self.ready = True
        elif transition.truncated:
            self.trajectory.clear()

    def learn(self):
        if not self.ready:
            return {}
        trajectory = self.trajectory
        states = self.encode_batch([t.state for t in trajectory])
        masks = torch.as_tensor(np.stack([t.action_mask for t in trajectory]), device=self.device)
        actions = torch.tensor([t.action for t in trajectory], device=self.device)
        returns = torch.as_tensor(discounted_returns([t.reward for t in trajectory], self.config.gamma),
                                  device=self.device)
        # Use gamma^t * G_t for the discounted-return objective. Do not backpropagate through G_t.
        weights = self.config.gamma ** torch.arange(len(trajectory), device=self.device)
        self.model.train()
        logits = self.model(states).masked_fill(~masks, -torch.inf)
        log_probs = Categorical(logits=logits).log_prob(actions)
        loss = -(weights * returns.detach() * log_probs).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("REINFORCE loss is not finite")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.updates += 1
        self.trajectory = []
        self.ready = False
        return {"loss": float(loss.item()), "updates": self.updates}

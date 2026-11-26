"""Student algorithm: Advantage Actor-Critic (A2C) with n-step returns.

Actor  = self.model  (policy logits, shape (B,4))
Critic = self.critic (state value V(s), shape (B,1)), built by build_model(config, output_dim=1)
Updates every N_STEPS environment steps, or when an episode ends/truncates.
"""
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from agents.base_agent import ActionResult, TorchAgent, read_checkpoint, ENVIRONMENT_ID
from models import build_model


class StudentAgent(TorchAgent):
    # A2C hyperparameters
    N_STEPS = 20          # update interval (ablation: 5 vs 20)
    VALUE_COEF = 0.5      # critic loss weight
    ENTROPY_COEF = 0.01   # exploration bonus (ablation: 0 vs 0.01)

    def __init__(self, config):
        super().__init__(config)
        self.critic = build_model(config, output_dim=1).to(self.device)
        params = list(self.model.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        self.buffer = []


    @torch.no_grad()
    def act(self, state, action_mask, training=False):
        mask = self.check_mask(action_mask)
        self.model.eval()
        logits = self.model(self.encode_batch([state]))[0]
        logits = logits.masked_fill(~torch.as_tensor(mask, device=self.device), -torch.inf)
        dist = Categorical(logits=logits)
        if training:
            p = dist.probs.cpu().numpy().astype(float)
            action = int(self.rng.choice(4, p=p / p.sum()))
        else:
            action = int(logits.argmax().item())
        log_prob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
        return ActionResult(action, {"log_prob": log_prob})


    def observe(self, transition):
        self.env_steps += 1
        self.buffer.append(transition)


    def learn(self):
        if not self.buffer:
            return {}
        last = self.buffer[-1]
        episode_over = last.terminated or last.truncated
        if len(self.buffer) < self.N_STEPS and not episode_over:
            return {}

        batch, self.buffer = self.buffer, []
        gamma = self.config.gamma

        # 1) bootstrap value: terminated -> 0, truncated/ongoing -> V(next_state)
        if last.terminated:
            bootstrap = 0.0
        else:
            with torch.no_grad():
                self.critic.eval()
                bootstrap = float(self.critic(self.encode_batch([last.next_state]))[0, 0].item())

        # 2) n-step returns, computed backwards (batch never spans two episodes)
        returns = np.zeros(len(batch), dtype=np.float32)
        future = bootstrap
        for i in reversed(range(len(batch))):
            future = float(batch[i].reward) + gamma * future
            returns[i] = future

        # 3) tensors
        states = self.encode_batch([t.state for t in batch])
        masks = torch.as_tensor(np.stack([t.action_mask for t in batch]), device=self.device)
        actions = torch.tensor([t.action for t in batch], device=self.device)
        returns = torch.as_tensor(returns, device=self.device)

        # 4) losses
        self.model.train()
        self.critic.train()
        logits = self.model(states).masked_fill(~masks, -torch.inf)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.critic(states).squeeze(-1)
        advantage = (returns - values).detach()

        policy_loss = -(log_probs * advantage).mean()
        value_loss = torch.nn.functional.smooth_l1_loss(values, returns)
        loss = policy_loss + self.VALUE_COEF * value_loss - self.ENTROPY_COEF * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError("A2C loss is not finite")

        self.optimizer.zero_grad()
        loss.backward()
        params = list(self.model.parameters()) + list(self.critic.parameters())
        torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
        self.optimizer.step()
        self.updates += 1
        return {"loss": float(loss.item()), "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()), "entropy": float(entropy.item()),
                "updates": self.updates}


    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "environment": ENVIRONMENT_ID,
                   "config": self.config.to_dict(),
                   "model": self.model.state_dict(),
                   "critic": self.critic.state_dict(),
                   "optimizer": self.optimizer.state_dict(),
                   "env_steps": self.env_steps, "updates": self.updates}
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def load(self, path):
        payload = read_checkpoint(path)
        old, new = dict(payload["config"]), self.config.to_dict()
        old.pop("device", None)
        new.pop("device", None)
        if old != new:
            raise ValueError("Use registry.load_agent to restore the checkpoint configuration automatically")
        self.model.load_state_dict(payload["model"])
        self.critic.load_state_dict(payload["critic"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.env_steps = payload["env_steps"]
        self.updates = payload["updates"]
        self.model.eval()
        self.critic.eval()
        return self
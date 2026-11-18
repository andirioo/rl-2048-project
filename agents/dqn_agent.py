"""Default DQN. STUDENT EDIT markers identify action selection, targets, and loss functions."""
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from agents.base_agent import ActionResult, TorchAgent
from replay import ReplayBuffer


class DQNAgent(TorchAgent):
    def __init__(self, config):
        super().__init__(config)
        self.target_model = deepcopy(self.model).eval()
        self.target_model.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, config.seed + 2)

    def epsilon(self):
        fraction = min(self.env_steps / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start)

    @torch.no_grad()
    def act(self, state, action_mask, training=False):
        # STUDENT EDIT D1: Replace exploration. Evaluation always disables epsilon exploration.
        mask = self.check_mask(action_mask)
        if training and self.rng.random() < self.epsilon():
            action = int(self.rng.choice(np.flatnonzero(mask)))
        else:
            self.model.eval()
            q = self.model(self.encode_batch([state]))[0]
            legal = torch.as_tensor(mask, device=self.device)
            action = int(q.masked_fill(~legal, -torch.inf).argmax().item())
        return ActionResult(action)

    def observe(self, transition):
        self.replay.push(transition)
        self.env_steps += 1

    @torch.no_grad()
    def compute_td_target(self, rewards, next_states, terminated, next_masks):
        """STUDENT EDIT D2: Replace target computation, for example with Double DQN.

        rewards/terminated: (B,); next_masks: bool (B, 4).
        An external truncation is not an MDP terminal state; use terminated only.
        """
        if torch.any((~terminated) & (~next_masks.any(dim=1))):
            raise ValueError("Nonterminal samples must have a legal next action")
        q_next = self.target_model(next_states).masked_fill(~next_masks, -torch.inf)
        best = q_next.max(dim=1).values
        # Set terminal values directly to zero; 0 * -inf would produce NaN.
        best = torch.where(terminated, torch.zeros_like(best), best)
        return rewards + self.config.gamma * best

    def compute_loss(self, predictions, targets):
        # STUDENT EDIT D3: Compute TD loss only for the actions actually taken.
        return F.smooth_l1_loss(predictions, targets)

    def learn(self):
        c = self.config
        if (self.env_steps < c.learning_starts or len(self.replay) < c.batch_size
                or self.env_steps % c.train_every != 0):
            return {}
        batch = self.replay.sample(c.batch_size)
        states = self.encode_batch([t.state for t in batch])
        next_states = self.encode_batch([t.next_state for t in batch])
        actions = torch.tensor([t.action for t in batch], device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        terminated = torch.tensor([t.terminated for t in batch], dtype=torch.bool, device=self.device)
        masks = torch.as_tensor(np.stack([t.next_action_mask for t in batch]), device=self.device)
        targets = self.compute_td_target(rewards, next_states, terminated, masks)
        self.model.train()
        predictions = self.model(states).gather(1, actions[:, None]).squeeze(1)
        loss = self.compute_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("DQN loss is not finite; check rewards and target computation")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % c.target_update_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return {"loss": float(loss.item()), "epsilon": self.epsilon(),
                "updates": self.updates}

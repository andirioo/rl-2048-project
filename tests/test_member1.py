import unittest
import torch
from torch import nn

from settings import Config
from agents.dqn_agent import DQNAgent


class Member1Tests(unittest.TestCase):
    def test_double_dqn_online_selects_target_evaluates(self):
        config = Config(network="mlp", hidden_units=8, batch_size=2,
                        double_dqn=True)
        agent = DQNAgent(config)

        class OnlineQ(nn.Module):
            def forward(self, x):
                return torch.tensor([1.0, 50.0, 9.0, 2.0]).repeat(len(x), 1)

        class TargetQ(nn.Module):
            def forward(self, x):
                return torch.tensor([20.0, 100.0, 4.0, 3.0]).repeat(len(x), 1)

        agent.model = OnlineQ()
        agent.target_model = TargetQ()
        states = torch.zeros(1, 32, 4, 4)
        rewards = torch.tensor([2.0])
        terminated = torch.tensor([False])
        masks = torch.tensor([[True, False, True, False]])
        target = agent.compute_td_target(rewards, states, terminated, masks)
        torch.testing.assert_close(target, torch.tensor([2.0 + .99 * 4.0]))

    def test_cosine_epsilon_has_correct_endpoints(self):
        config = Config(network="mlp", hidden_units=8, epsilon_start=1.0,
                        epsilon_end=0.05, epsilon_decay_steps=100,
                        epsilon_schedule="cosine")
        agent = DQNAgent(config)
        agent.env_steps = 0
        self.assertAlmostEqual(agent.epsilon(), 1.0)
        agent.env_steps = 100
        self.assertAlmostEqual(agent.epsilon(), 0.05)


if __name__ == "__main__":
    unittest.main()

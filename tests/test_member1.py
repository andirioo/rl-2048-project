import unittest
import numpy as np
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


class FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values)

    def forward(self, x):
        return self.values.repeat(len(x), 1)


def softmax_agent(values, **kwargs):
    agent = DQNAgent(Config(network="mlp", hidden_units=8, epsilon_schedule="softmax",
                            epsilon_decay_steps=100, **kwargs))
    agent.model = FixedQ(values)
    return agent


class Member1SoftmaxTests(unittest.TestCase):
    BOARD = np.array([[2, 0, 0, 0]] + [[0, 0, 0, 0]] * 3)

    def test_temperature_decays_and_no_random_moves(self):
        agent = softmax_agent([0.0, 0.0, 0.0, 0.0])
        self.assertEqual(agent.epsilon(), 0.0)
        self.assertAlmostEqual(agent.temperature(), 1.0)
        agent.env_steps = 100
        self.assertAlmostEqual(agent.temperature(), 0.05)

    def test_softmax_explores_early_and_is_nearly_greedy_late(self):
        agent = softmax_agent([10.0, 9.5, 0.0, 5.0])
        mask = np.ones(4, dtype=bool)
        early = [agent.act(self.BOARD, mask, training=True).action for _ in range(400)]
        self.assertEqual(len(set(early)), 4)
        agent.env_steps = 100
        late = [agent.act(self.BOARD, mask, training=True).action for _ in range(400)]
        self.assertNotIn(2, late)                         # clearly bad move disappears
        self.assertGreater(late.count(0), late.count(1))  # best move preferred
        self.assertGreater(late.count(1), 0)              # near-tie still explored

    def test_softmax_respects_mask_and_evaluation_is_greedy(self):
        agent = softmax_agent([9.0, 1.0, 9.0, 2.0])
        mask = np.array([False, True, False, True])
        actions = {agent.act(self.BOARD, mask, training=True).action for _ in range(200)}
        self.assertTrue(actions <= {1, 3})
        state = agent.rng.bit_generator.state
        self.assertEqual(agent.act(self.BOARD, mask, training=False).action, 3)
        self.assertEqual(agent.rng.bit_generator.state, state)  # evaluation uses no randomness

    def test_invalid_temperature_rejected(self):
        with self.assertRaises(ValueError):
            Config(epsilon_schedule="softmax", temperature_start=0.1, temperature_end=0.5).validate()


if __name__ == "__main__":
    unittest.main()

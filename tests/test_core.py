"""Run: python -m unittest discover -s tests -v"""
from copy import deepcopy
from itertools import product
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn

from env import Game2048, legal_action_mask, move_board
from encoders import encode_state
from settings import Config
from rewards import compute_reward
from models import build_model
from replay import ReplayBuffer
from agents.base_agent import Transition
from agents.dqn_agent import DQNAgent
from agents.reinforce_agent import ReinforceAgent, discounted_returns
from agents.registry import create_agent, load_agent
from baselines import BaselineAgent
from evaluate import evaluate_agent
from reporting import summarize

torch.set_num_threads(1)


def sample_transition(terminated=False, truncated=False, reward=4.0):
    state = np.zeros((4, 4), dtype=np.int64)
    state[0, 0] = 2
    mask = legal_action_mask(state)
    next_mask = np.zeros(4, dtype=bool) if terminated else mask.copy()
    return Transition(state.copy(), 2, reward, state.copy(), terminated, truncated,
                      mask, next_mask)


class EnvironmentTests(unittest.TestCase):
    def test_merge_once_and_scores(self):
        for row, expected, score in [([2, 2, 4, 0], [4, 4, 0, 0], 4),
                                     ([2, 2, 2, 2], [4, 4, 0, 0], 8),
                                     ([4, 4, 8, 8], [8, 16, 0, 0], 24),
                                     ([2, 0, 2, 2], [4, 2, 0, 0], 4)]:
            board = np.zeros((4, 4), dtype=np.int64)
            board[0] = row
            original = board.copy()
            after, gain, _ = move_board(board, 1)
            np.testing.assert_array_equal(after[0], expected)
            np.testing.assert_array_equal(board, original)
            self.assertEqual(gain, score)

    def test_all_row_patterns_in_four_directions(self):
        # Independent reference: consume adjacent pairs without calling the tested merge function.
        for values in product((0, 2, 4, 8), repeat=4):
            for action in range(4):
                board = np.zeros((4, 4), dtype=np.int64)
                if action in (0, 3):
                    board[:, 0] = values
                else:
                    board[0, :] = values
                sequence = list(values[::-1] if action in (2, 3) else values)
                remaining = [v for v in sequence if v]
                expected, score = [], 0
                while remaining:
                    value = remaining.pop(0)
                    if remaining and value == remaining[0]:
                        value += remaining.pop(0)
                        score += value
                    expected.append(value)
                expected += [0] * (4 - len(expected))
                if action in (2, 3):
                    expected.reverse()
                after, gain, _ = move_board(board, action)
                observed = after[:, 0] if action in (0, 3) else after[0]
                np.testing.assert_array_equal(observed, expected)
                self.assertEqual(gain, score)

    def test_invalid_action_does_not_spawn(self):
        env = Game2048(seed=2)
        board = np.zeros((4, 4), dtype=np.int64)
        board[0, 0] = 2
        env.reset(board=board)
        rng_before = deepcopy(env.rng.bit_generator.state)
        observed, reward, term, trunc, info = env.step(0)
        np.testing.assert_array_equal(observed, board)
        self.assertEqual(env.rng.bit_generator.state, rng_before)
        self.assertEqual(reward, 0)
        self.assertFalse(info["moved"] or term or trunc)

    def test_terminal_after_new_tile(self):
        board = np.array([[2, 4, 8, 16], [4, 8, 16, 32],
                          [8, 16, 32, 64], [0, 2, 4, 8]], dtype=np.int64)
        env = Game2048(seed=3)
        env.reset(board=board)
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated or info["action_mask"].any())
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_continue_after_2048_and_truncation(self):
        env = Game2048(seed=2, max_steps=1)
        board = np.zeros((4, 4), dtype=np.int64)
        board[0, :2] = 1024
        env.reset(board=board)
        _, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(reward, 2048)
        self.assertEqual(info["max_tile"], 2048)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_reset_and_reproducible_environment(self):
        a, b = Game2048(seed=9), Game2048(seed=9)
        s1, i1 = a.reset()
        s2, _ = b.reset()
        self.assertEqual(np.count_nonzero(s1), 2)
        for _ in range(20):
            np.testing.assert_array_equal(s1, s2)
            action = int(np.flatnonzero(i1["action_mask"])[0])
            s1, _, term, _, i1 = a.step(action)
            s2, _, _, _, _ = b.step(action)
            if term:
                break

    def test_full_board_still_mergeable_at_edge(self):
        board = np.array([[2, 4, 8, 16], [4, 8, 16, 32],
                          [8, 16, 32, 64], [16, 32, 64, 64]])
        self.assertTrue(legal_action_mask(board).any())
        board[3, 3] = 128
        self.assertFalse(legal_action_mask(board).any())


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = Config(network="mlp", hidden_units=16, batch_size=2,
                             learning_starts=2, train_every=1, target_update_every=1)

    def test_encodings_and_model_shapes(self):
        board = np.zeros((4, 4), dtype=np.int64)
        board[0, 0] = 65536
        encoded = encode_state(board)
        self.assertEqual(encoded.shape, (32, 4, 4))
        np.testing.assert_array_equal(encoded.sum(axis=0), np.ones((4, 4)))
        self.assertEqual(encoded[16, 0, 0], 1)
        self.assertEqual(encode_state(board, "log2").shape, (1, 4, 4))
        with self.assertRaises(ValueError):
            encode_state(board, channels=16)
        for network in ("mlp", "cnn"):
            for encoding in ("one_hot", "log2"):
                c = Config(network=network, encoding=encoding, conv_depth=8, hidden_units=16)
                m = build_model(c)
                x = torch.tensor(np.stack([encode_state(board, encoding)] * 2))
                self.assertEqual(m(x).shape, (2, 4))

    def test_dqn_targets_legal_mask_terminal_and_truncation(self):
        agent = DQNAgent(self.config)
        class ConstantQ(nn.Module):
            def forward(self, x):
                return torch.tensor([1., 100., 7., 4.]).repeat(len(x), 1)
        agent.target_model = ConstantQ()
        states = torch.zeros(3, 32, 4, 4)
        rewards = torch.tensor([2., 3., 4.])
        terminal = torch.tensor([False, True, False])
        masks = torch.tensor([[True, False, True, False], [False]*4,
                              [True, False, False, True]])
        target = agent.compute_td_target(rewards, states, terminal, masks)
        # The third sample still bootstraps if time-truncated but not naturally terminated.
        expected = torch.tensor([2 + .99*7, 3, 4 + .99*4])
        torch.testing.assert_close(target, expected)
        self.assertTrue(torch.isfinite(target).all())
        self.assertFalse(target.requires_grad)

    def test_epsilon_floor_and_legal_actions(self):
        agent = DQNAgent(self.config)
        t = sample_transition()
        agent.env_steps = 10**9
        self.assertAlmostEqual(agent.epsilon(), self.config.epsilon_end)
        for training in (False, True):
            for _ in range(20):
                action = agent.act(t.state, t.action_mask, training).action
                self.assertTrue(t.action_mask[action])
        self.assertEqual(agent.env_steps, 10**9)

    def test_replay_copies_and_is_bounded(self):
        memory = ReplayBuffer(2)
        t = sample_transition()
        memory.push(t)
        t.state[0, 0] = 64
        self.assertEqual(memory.data[0].state[0, 0], 2)
        memory.push(sample_transition(reward=2))
        memory.push(sample_transition(reward=3))
        self.assertEqual(len(memory), 2)
        self.assertEqual({t.reward for t in memory.sample(2)}, {2, 3})

    def test_dqn_updates_and_target_sync(self):
        agent = DQNAgent(self.config)
        before = deepcopy(agent.model.state_dict())
        agent.observe(sample_transition())
        agent.observe(sample_transition(terminated=True))
        result = agent.learn()
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertEqual(agent.updates, 1)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in agent.model.state_dict().items()))
        for a, b in zip(agent.model.parameters(), agent.target_model.parameters()):
            torch.testing.assert_close(a, b)

    def test_reinforce_returns_and_complete_episode_update(self):
        np.testing.assert_allclose(discounted_returns([1, 2, 3], .5), [2.75, 3.5, 3])
        agent = ReinforceAgent(self.config)
        agent.observe(sample_transition())
        self.assertEqual(agent.learn(), {})
        agent.observe(sample_transition(terminated=True))
        before = deepcopy(agent.model.state_dict())
        self.assertTrue(np.isfinite(agent.learn()["loss"]))
        self.assertEqual(agent.updates, 1)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in agent.model.state_dict().items()))
        agent.observe(sample_transition(truncated=True))
        self.assertEqual(agent.learn(), {})
        self.assertEqual(len(agent.trajectory), 0)

    def test_checkpoint_roundtrip_all_algorithms(self):
        for algorithm in ("dqn", "reinforce", "student"):
            for network in ("mlp", "cnn"):
                with self.subTest(algorithm=algorithm, network=network), tempfile.TemporaryDirectory() as d:
                    c = Config(algorithm=algorithm, network=network, conv_depth=4, hidden_units=8)
                    a = create_agent(c)
                    a.env_steps = 123
                    path = Path(d) / "model.pt"
                    a.save(path)
                    b = load_agent(path)
                    x = a.encode_batch([sample_transition().state])
                    with torch.no_grad():
                        torch.testing.assert_close(a.model(x), b.model(x), rtol=0, atol=0)
                    self.assertEqual(b.env_steps, 123)
                    self.assertEqual(b.config.algorithm, algorithm)

    def test_reward_modes_do_not_change_game_score(self):
        state = np.zeros((4, 4), dtype=np.int64)
        state[0, :2] = 2
        after, gain, merges = move_board(state, 1)
        info = {"score_delta": gain, "score": gain, "merges": merges}
        self.assertEqual(compute_reward(state, after, info, Config()), 4)
        self.assertAlmostEqual(compute_reward(state, after, info, Config(reward="merge_bonus")), 1.2)
        self.assertEqual(info["score"], 4)


class EvaluationTests(unittest.TestCase):
    def test_statistics_exact_counts_and_std(self):
        records = [{"score": score, "max_tile": tile, "terminated": True, "truncated": False}
                   for score, tile in zip([0, 100, 200, 300], [256, 512, 1024, 2048])]
        summary = summarize(records, "test")
        self.assertEqual(summary["mean_score"], 150)
        self.assertAlmostEqual(summary["score_std"], np.std([0, 100, 200, 300], ddof=1))
        self.assertEqual([summary[f"rate_{t}"] for t in (512, 1024, 2048)], [.75, .5, .25])
        records.append({"score": 100000, "max_tile": 4096, "terminated": False, "truncated": True})
        incomplete = summarize(records, "test")
        self.assertEqual(incomplete["mean_score"], 150)
        self.assertFalse(incomplete["evaluation_complete"])
        self.assertEqual(incomplete["episodes_truncated"], 1)

    def test_no_completed_games_and_single_game_std(self):
        s = summarize([{"score": 0, "max_tile": 2, "terminated": False, "truncated": True}], "test")
        self.assertIsNone(s["mean_score"])
        self.assertIsNone(s["rate_512"])
        s = summarize([{"score": 4, "max_tile": 4, "terminated": True, "truncated": False}], "test")
        self.assertIsNone(s["score_std"])

    def test_evaluation_never_updates_agent(self):
        agent = DQNAgent(Config(network="mlp", hidden_units=8))
        before = deepcopy(agent.model.state_dict())
        def forbidden(*args):
            raise AssertionError("evaluation must not train")
        agent.observe = forbidden
        agent.learn = forbidden
        records = evaluate_agent(agent, episodes=2, max_steps=1000, progress=False)
        self.assertEqual(len(records), 2)
        self.assertEqual(agent.env_steps, 0)
        self.assertEqual(agent.updates, 0)
        self.assertEqual(len(agent.replay), 0)
        for name, parameter in agent.model.state_dict().items():
            torch.testing.assert_close(before[name], parameter, rtol=0, atol=0)

    def test_baselines_and_truncation_reporting(self):
        for strategy in ("random", "greedy"):
            rows = evaluate_agent(BaselineAgent(strategy), episodes=2, max_steps=1, progress=False)
            self.assertTrue(all(r["truncated"] and not r["terminated"] for r in rows))
            self.assertEqual(summarize(rows, strategy)["episodes_completed"], 0)


if __name__ == "__main__":
    unittest.main()

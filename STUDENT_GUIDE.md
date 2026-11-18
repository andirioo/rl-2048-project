# Student Modification Guide

Run the default configuration and tests before making changes. Change one factor at a time, use a new output directory for each experiment, and preserve the configuration and results. Evaluate each final model over 100 episodes using the shared `evaluate.py` script.

## A. Replace the State Encoding

Edit `encode_state()` in `encoders.py`.

- Input: an integer board of shape `(4,4)`.
- Output: a `float32` array of shape `(C,4,4)`.
- When adding an encoding, update `input_channels()` and the encoding validation in `settings.py`.
- If the encoding discards board information, account for that loss when interpreting results. It can be an experimental question in its own right.

The built-in choices are `one_hot` and `log2`. The model automatically uses the corresponding number of input channels.

## B. Replace the Network while Keeping the Algorithm

Edit `build_model()` in `models.py`.

Both built-in networks take `(B,C,4,4)` and return `(B,4)`. DQN interprets the four outputs as Q-values. REINFORCE interprets them as action logits and uses `Categorical` to construct a probability distribution.

```python
class MyNetwork(nn.Module):
    def __init__(self, channels, output_dim=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 16, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.layers(x)
```

Register the class under a name such as `my_network` in `build_model()`, then select that name in the configuration. A Q-network should not end in softmax. The supplied policy-gradient implementation also expects raw logits and constructs its own distribution.

To study only the effect of network architecture, copy the same configuration and change only `network`. The supplied `dqn_mlp.json` changes both architecture and encoding; it is a runnable example, not a controlled comparison.

## C. Change the Reward

Edit `compute_reward()` in `rewards.py`.

Available inputs include the board before the move, the board after a new tile has spawned, `info['score_delta']`, and `info['merges']`. The default `score` reward is the score increment for the current move. The `merge_bonus` example uses the merge count plus `0.1*log2(new_max)` when the maximum tile increases.

Register any new reward name in the configuration validation. The environment independently maintains the raw game score; do not overwrite it with the training reward. This allows models trained with different rewards to share the same evaluation metrics.

## D. Modify One Component of DQN

Edit `agents/dqn_agent.py`.

| Method | Possible experiments | Required behavior |
|---|---|---|
| `act()` | Epsilon schedules or other exploration strategies | Mask invalid actions and disable training exploration during evaluation |
| `compute_td_target()` | DQN, Double DQN, or another target | Do not bootstrap at terminal states; use legal next actions and detached targets |
| `compute_loss()` | Huber loss, MSE, or weighted losses | Compare the Q-value of the action taken with its target |
| `learn()` | Update timing or sampling strategies | Update parameters only during training |

The DQN target is:

```text
Terminal state:  y = reward
Otherwise:       y = reward + gamma * max_{legal action a} Q_target(next_state, a)
```

For Double DQN, use the online network to select a legal action and the target network to evaluate it. Handle terminal states explicitly. An all-false terminal action mask must not lead to `0 * (-inf)`.

## E. Replace the Entire Algorithm

Edit `agents/student_agent.py`. This entry point inherits REINFORCE by default and can run immediately. It does not depend on `DQNAgent` or `replay.py`.

1. Copy `configs/student.json` and choose your own output directory.
2. Modify `act()`, `observe()`, and `learn()` in `StudentAgent`.
3. If you add networks or optimizers, define `__init__()` and update `save()` and `load()` accordingly.
4. Continue using the shared `train.py` and `evaluate.py` scripts.

All algorithms use this interface:

```python
decision = agent.act(state, action_mask, training=True)
# decision is ActionResult(action: int, info: dict)

agent.observe(transition)
metrics = agent.learn()  # Return {} if no update occurs, otherwise numeric metrics.

agent.save(path)
agent.load(path)
```

The `Transition` fields are:

| Field | Meaning |
|---|---|
| `state`, `next_state` | Raw boards, each shaped `(4,4)` |
| `action` | Action ID from 0 to 3 |
| `reward` | Training reward computed by the configured reward function |
| `terminated` | Natural game termination |
| `truncated` | An external budget or step limit ended the trajectory |
| `action_mask`, `next_action_mask` | Legal action masks for both states, each bool `(4,)` |
| `action_info` | Additional information returned by `act()` |

The trainer calls `observe()` and `learn()` after every environment step. Each algorithm controls its own storage and update schedule; it need not update on every call.

- **DQN:** `observe()` adds transitions to replay. `learn()` updates when the sample count and update interval allow it.
- **REINFORCE:** `observe()` collects a complete episode. `learn()` updates once after natural termination.
- **PPO or other rollout algorithms:** store old log-probabilities and value estimates in `ActionResult.info`, collect a fixed-length rollout, and compute advantages and multiple update passes inside `learn()`. The algorithm manages its actor, critic, and checkpoints.

PPO is an extension example, not a built-in implementation. Adding a configuration name alone does not implement it. Register any new algorithm class in the `AGENTS` dictionary in `agents/registry.py`.

Registered classes receive a `Config` object in their constructor. To retain automatic loading, keep these top-level checkpoint fields: `format_version=1`, `environment=ENVIRONMENT_ID`, and `config=config.to_dict()`. Additional networks can use separate fields. `load_agent()` restores the configuration, constructs the registered algorithm, and calls its `load()` method. Follow `TorchAgent.save/load` when adding modules, and preserve the checkpoint format convention. The `updates` counter is optional diagnostic information.

## F. Distinguish Termination from Truncation

A naturally terminated game has no future game rewards. A training budget may stop while the current board still has legal moves.

- DQN uses `terminated` to mask future values. A budget-truncated sample can still bootstrap.
- The supplied REINFORCE implementation has no value network to estimate the missing tail return, so it discards the final incomplete episode.
- A replacement algorithm must handle both flags correctly. Do not merge them indiscriminately when computing learning targets.

## G. Check a Replacement Algorithm

- `act()` always returns a legal action and does not update parameters or collect training data during evaluation.
- Network shapes, action IDs, and action masks agree.
- Targets and losses remain finite.
- Saving and loading preserves outputs for the same board, including any additional networks such as a critic.
- `evaluate.py` completes 100 episodes and exports all required statistics.

The game environment, evaluation loop, and statistics can stay shared. Retrain after changing the algorithm; do not reuse another algorithm's checkpoint.

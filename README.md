# 2048 Reinforcement Learning Course Project

A working starter project for an undergraduate reinforcement learning course. It includes a shared 2048 environment, DQN, REINFORCE, random and greedy baselines, and independent evaluation over 100 episodes per model by default. Students can replace the network, state encoding, reward, learning target, or entire algorithm.

The default training budget provides a starting point for experiments; it does not guarantee reaching 2048. Files under `outputs/validation_*` are short-run integration examples, not final performance benchmarks. Multiple training seeds are not required.

## 1. Installation

Use Python 3.10 or later. This project was tested on Windows with Python 3.12 and a CPU. Open a terminal in the project directory and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands install the CPU version of PyTorch. For GPU training, install the appropriate version using the official PyTorch instructions and add `--device cuda` to the training command.

On Linux or macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. In the examples below, `python` means the Python executable in this project's virtual environment. On Windows, you can always use the full executable path without activating the environment.

`requirements.txt` specifies compatible version ranges. `requirements-tested.txt` records the direct dependency versions used for validation.

## 2. First Run

Check the environment and core algorithm logic:

```bash
python -m unittest discover -s tests -v
```

The package includes short training runs and evaluation examples. Independently check the saved statistics for five sets of 100 episodes and the exit code for truncated evaluations:

```bash
python tests/verify_outputs.py
```

These examples use 3000 DQN training steps, 1500 REINFORCE training steps, 500 student-agent training steps, and two baselines requiring no training. The different budgets exercise the available execution paths; they do not provide a fair algorithm comparison.

Evaluate the baselines:

```bash
python evaluate.py --baseline random --output outputs/random_eval
python evaluate.py --baseline greedy --output outputs/greedy_eval
```

The random baseline samples uniformly from legal actions. The greedy baseline selects the legal action with the highest immediate merge score and breaks ties uniformly at random. Both use the same environment and metrics as the learned agents.

## 3. Train and Evaluate DQN

The default configuration uses a CNN, one-hot state encoding, and the raw score increment as the reward:

```bash
python train.py --config config.json
python evaluate.py --checkpoint outputs/dqn_cnn/final.pt --output outputs/dqn_cnn_eval
```

The default budget is 20000 environment steps, followed by 100 evaluation episodes. Override the training budget for a quick installation and workflow check:

```bash
python train.py --steps 3000 --output outputs/my_quick_run
python evaluate.py --checkpoint outputs/my_quick_run/final.pt --episodes 100 --output outputs/my_quick_eval
```

Output directories must be empty. Use a new `--output` directory for each experiment to preserve previous results. If no evaluation output directory is specified, the evaluator creates a new directory with a timestamp.

Display a single game in the terminal:

```bash
python evaluate.py --checkpoint outputs/dqn_cnn/final.pt --episodes 1 --render --delay 0.05
```

The demonstration uses the same environment and model as evaluation, and it supports tiles larger than 2048.

## 4. Replace the Network or Algorithm

The following configurations are ready to run:

```bash
python train.py --config configs/dqn_mlp.json
python train.py --config configs/reinforce.json
python train.py --config configs/student.json
```

- `dqn_mlp.json`: DQN with an MLP and log2 encoding. To isolate the effect of network architecture, create a separate configuration that changes only `network` and keeps the encoding and other settings fixed.
- `reinforce.json`: A complete REINFORCE policy gradient algorithm using whole episodes, without replay or Q-learning.
- `student.json`: Runs `agents/student_agent.py`. Its initial behavior matches REINFORCE, so students can run it immediately and then replace its components.

The evaluator automatically restores the algorithm, network, and encoding from the checkpoint:

```bash
python evaluate.py --checkpoint outputs/reinforce/final.pt --output outputs/reinforce_eval
python evaluate.py --checkpoint outputs/student/final.pt --output outputs/student_eval
```

See [STUDENT_GUIDE.md](STUDENT_GUIDE.md) for modification instructions. Search the source for `STUDENT EDIT` to locate the extension points.

## 5. Evaluation Results

Each model is evaluated over 100 episodes by default. Evaluation never updates model parameters. DQN disables epsilon exploration, and REINFORCE selects the legal action with the highest probability.

| File | Contents |
|---|---|
| `episodes.csv` | Episode index, environment seed, raw game score, maximum tile, steps, and termination/truncation flags |
| `summary.csv` / `summary.json` | Mean score, sample standard deviation, and counts and rates for reaching 512, 1024, and 2048 |
| `evaluation.png` | Score distribution and tile achievement rates |
| `evaluation_config.json` | Evaluation settings and the model's original training configuration |

Metric definitions:

- **Mean score:** the arithmetic mean of the final raw game scores from completed episodes.
- **Score standard deviation:** the sample standard deviation, with denominator `n-1`. It is undefined for a single episode and is saved as a null or empty value.
- **512 achievement rate:** the number of episodes whose maximum tile is at least 512, divided by the number of completed episodes. The definitions for 1024 and 2048 are identical.
- Achievement rates overlap: an episode reaching 2048 also counts toward the 512 and 1024 rates.
- CSV rates range from 0 to 1. The terminal and plots display percentages.
- Evaluation always uses the raw game score, regardless of the training reward function.

Training and evaluation use separate random-number settings. Evaluation starts at environment seed 2026 by default and generates a different random game for each episode. These are 100 evaluation episodes for one trained model, not repeated training runs with multiple seeds. Using the same evaluation settings across methods helps standardize conditions, although different actions still lead to different trajectories.

A default limit of 20000 steps protects against excessively long evaluation episodes. Reaching this limit sets `truncated`, which differs from natural termination. Truncated episodes remain in `episodes.csv` but are excluded from completed-episode metrics. The summary sets `evaluation_complete=false`, and the CLI exits with code 2. Increase `--max-steps` and rerun before reporting a complete 100-episode evaluation. If no episode completes, the metrics are null.

## 6. Training Outputs

| File | Contents |
|---|---|
| `config.json` | Complete configuration, including default values |
| `train_episodes.csv` | Raw score, total training reward, episode length, and termination flags |
| `train_updates.csv` | Loss, epsilon when applicable, and additional algorithm metrics for each update |
| `training.png` | Raw score, training reward, and loss curves against environment steps |
| `step_*.pt` / `final.pt` | Periodic checkpoints and the final model |
| `run_info.json` | Environment steps, completed/truncated episodes, update count, elapsed time, and versions |

The last training episode may be incomplete when the step budget is reached. It is explicitly marked as truncated. DQN still bootstraps from a nonterminal next state. REINFORCE updates only from complete episodes and discards the unfinished final trajectory.

Checkpoints contain the configuration, model, optimizer, and training counters. DQN also saves its target network. They support evaluation and demonstrations, but exact training resumption is not implemented: replay data, trajectories, environment state, and random-number states are not fully saved. After changing the architecture or algorithm, retrain using a new output directory.

## 7. Game Rules and Configuration

- The board is 4x4. Action IDs are fixed: `0=up, 1=left, 2=right, 3=down`.
- A game begins with two tiles. After each valid move, a new tile appears in an empty cell: 2 with probability 90%, or 4 with probability 10%.
- A tile can merge at most once per move. The score increases by the sum of the values of the newly merged tiles.
- Invalid moves do not spawn tiles. Agents must respect the legal action mask.
- Play continues beyond 2048 and ends when no legal action remains.

Common settings:

| Setting | Meaning |
|---|---|
| `algorithm` | `dqn`, `reinforce`, or `student` |
| `network` | `cnn` or `mlp` |
| `encoding` | `one_hot` or `log2` |
| `reward` | `score` or `merge_bonus` |
| `reward_scale` | Training reward multiplier, default 1; does not change the raw game score |
| `gamma` | Discount factor |
| `total_steps` | Training budget in environment steps |
| `learning_starts` / `train_every` | DQN learning start step / update interval |
| `target_update_every` | Number of DQN optimizer updates between target-network synchronizations |
| `epsilon_*` | DQN exploration start, floor, and decay duration in environment steps |
| `evaluation_episodes` | Default evaluation episode count, initially 100 |

`settings.py` defines all defaults. Configuration files can specify only the values they change. Unknown fields raise an error.

The CNN retains the original project's horizontal/vertical convolution and six-branch feature concatenation design. It uses 32 convolution channels and a 128-unit fully connected layer by default to reduce classroom runtime. The MLP has two hidden layers. One-hot encoding uses 32 channels for empty cells and powers from 2^1 through 2^31; values beyond this range raise an explicit error. Log2 encoding maps empty cells to 0 and other tiles to `log2(tile)/32`.

## 8. Source Map

```text
env.py                    Shared game rules for training and evaluation
encoders.py               State encoding extension point
models.py                 Network architecture extension point
rewards.py                Training reward extension point
agents/base_agent.py      Algorithm interface, transitions, checkpoint helpers
agents/dqn_agent.py        DQN action selection, targets, and loss
agents/reinforce_agent.py  Complete alternative algorithm
agents/student_agent.py   Entry point for replacing the entire algorithm
agents/registry.py        Algorithm registration and automatic loading
replay.py                 DQN replay buffer
train.py                  Shared training loop
evaluate.py               Shared evaluation and terminal demonstrations
reporting.py              Statistics and plots
baselines.py              Random and immediate-score greedy baselines
tests/test_core.py        Correctness checks
```

## 9. References and Attribution

This teaching implementation was rewritten from the supplied [2048 project by Navjinder Virdee](https://github.com/navjindervirdee/2048-deep-reinforcement-learning), retaining its network design ideas. The original MIT notice is preserved in [LICENSE](LICENSE). CSV weights from the original notebooks are not compatible with this implementation; retrain the model.

- [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html): replay, target networks, and terminal-state handling.
- [PyTorch distributions documentation](https://docs.pytorch.org/docs/stable/distributions.html): categorical policy distributions.
- [Gymnasium termination and truncation explanation](https://farama.org/Gymnasium-Terminated-Truncated-Step-API): why an external time limit differs from natural termination.

The current delivery covers the starter code. The supplied presentation and original archive have not been modified.

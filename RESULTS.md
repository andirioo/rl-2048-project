# Team results

Every configuration in `configs/` was trained with 3 training seeds (42, 7, 123) and evaluated with
`evaluate.py` on the same 100 games (evaluation seed 2026, greedy play). Each config file is unchanged except
`seed` and `total_steps`.

Simple baselines on the same 100 games: random moves **1,041**, greedy (best immediate merge) **2,073**.

## 20k training moves (course default)

| Config | Who | What | Mean score (3 seeds) | Seed range | 512 reached | 1024 reached |
|---|---|---|---|---|---|---|
| `configs/dqn_mlp.json` | Sandi John | DQN, MLP, log2 input | **2,281** | 2,209–2,426 | 2.7% | 0.0% |
| `configs/strategic_ddqn_mlp.json` | Sandi John + Kay Zin | Double DQN + cosine ε + strategic reward, MLP | **2,198** | 1,806–2,667 | 5.7% | 0.0% |
| `configs/strategic_dqn.json` | Sandi John + Kay Zin | Double DQN + cosine ε + strategic reward | **2,133** | 1,698–2,487 | 5.7% | 0.0% |
| `configs/student.json` | Uihyeon Kim | A2C, MLP, log2 input, reward × 0.01 | **1,994** | 794–2,980 | 6.0% | 0.0% |
| `configs/reinforce.json` | starter | REINFORCE, MLP, log2 input | **1,827** | 786–2,409 | 2.7% | 0.0% |
| `configs/ddqn_softmax.json` | Kay Zin | Double DQN + softmax exploration | **1,725** | 1,626–1,842 | 0.3% | 0.0% |
| `configs/dqn_standard.json` | starter | DQN, CNN, one-hot input, score reward | **1,705** | 1,350–1,970 | 0.7% | 0.0% |
| `configs/ddqn_linear.json` | Kay Zin | Double DQN | **1,680** | 1,535–1,829 | 0.7% | 0.0% |
| `configs/ddqn_cosine.json` | Kay Zin | Double DQN + cosine ε | **1,613** | 1,498–1,706 | 0.0% | 0.0% |
| `configs/student_strategic.json` | Uihyeon Kim + Sandi John | A2C + strategic reward, CNN, one-hot input | **1,362** | 671–2,070 | 1.7% | 0.0% |

## 100k training moves

| Config | Who | What | Mean score (3 seeds) | Seed range | 512 reached | 1024 reached |
|---|---|---|---|---|---|---|
| `configs/strategic_ddqn_softmax.json` | Kay Zin + Sandi John | Double DQN + softmax + strategic reward | **4,229** | 4,063–4,467 | 33.0% | 1.0% |
| `configs/ddqn_softmax.json` | Kay Zin | Double DQN + softmax exploration | **3,364** | 2,347–4,300 | 22.7% | 0.0% |
| `configs/strategic_dqn.json` | Sandi John + Kay Zin | Double DQN + cosine ε + strategic reward | **3,356** | 3,058–3,898 | 18.0% | 0.7% |
| `configs/student_strategic.json` | Uihyeon Kim + Sandi John | A2C + strategic reward, CNN, one-hot input | **2,635** | 2,414–2,874 | 7.7% | 0.0% |
| `configs/student.json` | Uihyeon Kim | A2C, MLP, log2 input, reward × 0.01 | **2,411** | 2,235–2,592 | 3.7% | 0.0% |
| `configs/dqn_mlp.json` | Sandi John | DQN, MLP, log2 input | **2,173** | 2,117–2,207 | 2.7% | 0.0% |
| `configs/strategic_ddqn_mlp.json` | Sandi John + Kay Zin | Double DQN + cosine ε + strategic reward, MLP | **2,140** | 1,484–2,489 | 5.3% | 0.0% |
| `configs/dqn_standard.json` | starter | DQN, CNN, one-hot input, score reward | **1,832** | 1,701–1,921 | 1.0% | 0.0% |
| `configs/ddqn_cosine.json` | Kay Zin | Double DQN + cosine ε | **1,816** | 1,681–1,926 | 1.7% | 0.0% |
| `configs/ddqn_linear.json` | Kay Zin | Double DQN | **1,671** | 1,624–1,719 | 0.7% | 0.0% |
| `configs/reinforce.json` | starter | REINFORCE, MLP, log2 input | **1,384** | 871–2,411 | 1.0% | 0.0% |

## Q-value check (100k moves, `measure_q_bias.py`)

Predicted Q(s, a) for the move played vs. the discounted reward actually received afterwards, on the 100 test games:

| Config | Mean predicted Q | Mean real return | (predicted − real) / real |
|---|---|---|---|
| `configs/dqn_standard.json` | 300 | 637 | -53% |
| `configs/ddqn_linear.json` | 293 | 597 | -51% |

Both networks underestimate, so Double DQN (which only lowers estimates) has nothing to correct in 2048.

## Reproduce one run

```bash
python train.py --config configs/strategic_ddqn_softmax.json --steps 100000 --output outputs/combo_100k
python evaluate.py --checkpoint outputs/combo_100k/final.pt --output outputs/combo_100k_eval
python measure_q_bias.py --checkpoint outputs/combo_100k/final.pt --output outputs/combo_100k_qbias
```

For another seed, copy the config and change `"seed"`. Numbers can differ slightly on another machine or
PyTorch version.

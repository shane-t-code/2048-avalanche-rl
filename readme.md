# 2048 Avalanche: N-Tuple TD Learning + Expectimax

A reinforcement-learning agent for **2048 Avalanche**, a harder variant of 2048 where merges cascade. The agent pairs a 500M-parameter n-tuple value network, trained by temporal-difference learning, with an expectimax search over the random tile spawns. It runs entirely on a laptop CPU with no GPU.

**Best game: 16384 tile, 309,948 points in 8,847 moves.** The game is fully reproducible from its seed (see [Reproduce the best game](#reproduce-the-best-game)).

<p align="center"><i>Built for the 2026 MIT Beaver Works Summer Institute (BWSI) Serious Games with AI course, on top of the course-provided game environment and baselines.</i></p>

---

## Results

| Agent | Highest tile | Best score | Avg score |
|---|---|---|---|
| Course baseline: greedy (down/left) | 256 | 3,314 | — |
| Course baseline: tabular Q-learning | 512 | 3,662 | — |
| Course baseline: Deep Q-Network | 256 | 2,396 | — |
| **This agent: n-tuple TD + expectimax** | **16384** | **309,948** | **138,116** (depth 3) |

That is **~85× the best baseline score** and a highest tile **5 doublings** above the best baseline.

**Search depth vs. strength** (3.03M-game network, 100 games per depth, +1 ply when ≤4 cells are empty):

| Search depth | P(tile ≥ 8192) | Avg score | Time / game |
|---|---|---|---|
| 1 | 70% | 111,585 | — |
| 2 | 77% | 122,680 | 2.0 s |
| 3 | 87% | 138,116 | 27.8 s |

- **Depth 3 is a ~5× multiplier on reaching 16384.** Over 3,600 games, 1.25% reached 16384 at depth 3, vs. 0.26% at depth 2.
- **Learning curve.** Temporal-coherence + 3-stage training (3.03M → 5.53M games) raised greedy P(≥8192) from 31.5% to 48.9%, and greedy average score from 76,357 to 96,013.

## Why this game is hard

2048 Avalanche differs from classic 2048 in three ways:

1. **Cascading merges.** A merged tile can keep merging in the same move, so one swipe can jump a tile up two levels (four 16384s → one 65536).
2. **50/50 spawns of 2 vs. 4** (classic 2048 uses 90/10). Boards are noisier, and good positions are harder to hold.
3. **Every turn spawns a tile, even a move that changed nothing,** and that move also costs 10 points. Wasted moves actively fill the board.

The provided DQN baseline averages negative scores because it repeatedly picks illegal moves. Masking illegal actions was mandatory from the start.

## How it works

### 1. Bitboard engine (~185× faster than the reference environment)
- The whole board packs into one 64-bit integer: 16 cells × 4 bits, each holding a tile exponent.
- All 65,536 possible rows are pre-slid into lookup tables. A move is 4 table lookups plus a bit-transpose.
- Engine code is JIT-compiled with **Numba**: **3.76M decisions/sec** (value lookups included), vs. ~20k steps/sec for the original environment.
- **Parity-tested against the original environment** on hand-traced cascade cases, randomized stress boards, and full games run move-by-move in lockstep. The engine fails loudly instead of overflowing past 32768.

### 2. N-tuple value network
- **10 hand-designed 6-cell patterns × 8 board symmetries = 80 lookups per board evaluation.**
- Each pattern indexes a table of 16⁶ ≈ 16.8M weights: **168M weights per stage, 503M across 3 stages.**
- Learning is **afterstate TD(0)**, following Szubert & Jaśkowski. The agent learns the value of the board *after* its move and *before* the random spawn. That separates its own decisions from the game's randomness.
- **Temporal coherence (TC):** each weight gets its own adaptive learning rate, |Σ errors| / Σ |errors|. Weights with consistently signed errors keep learning fast; weights with noisy errors settle down.
- **Multi-stage networks:** separate weight sets take over when the largest tile reaches 8192 and 16384. Late-game boards then get dedicated capacity instead of sharing weights with the opening.
- Training ran on a single CPU core at **~490k decisions/sec** (~2.4M games/hour early on), for **5.6M training games** in total.

### 3. Expectimax search
- **Max nodes** choose a legal move. **Chance nodes** average over every empty cell × {2, 4} at equal probability. **Leaves** use the n-tuple value.
- A **transposition table** (open-addressed, generation-stamped) caches positions reached through different move orders. Tests check that results are identical with the table on and off.
- **Adaptive depth** adds a ply when the board is nearly full. **Chance-node capping** samples a strided subset of empty cells on open boards.
- **Depth 1 is test-enforced to equal the greedy policy**, and the full search is checked against an independent reference implementation.

### 4. Evaluation infrastructure
- **Checkpoint tournament:** ranks every saved checkpoint by fast greedy play, then re-runs the top 6 under expectimax across a 6-process pool. Weight tables live in **shared memory**, so ~0.6 GB of weights load once instead of once per worker. In the final tournament, **all 6 finalists reached 16384.**
- **Deterministic per-game seeding:** every game is reproducible from `(run_seed, game_number)` alone, so any result can be replayed and verified in the GUI.

## A closed-form score identity

The score of any game turns out to depend only on its final board, not on how it was played:

$$\text{score} = \sum_{t} (\log_2 t - 1) \cdot t - 4 \cdot N_{4} - 10 \cdot N_{\text{no-op}}$$

where the sum runs over the tiles on the final board, $N_{4}$ is the number of 4-tiles spawned, and $N_{\text{no-op}}$ is the number of moves that changed nothing.

The identity holds because merges conserve the sum of tiles on the board. It is verified exactly on real games. Uses:

- **Plausibility checks:** any claimed (tile, score, turns) result can be checked for consistency.
- **Turn-cap ceiling:** a 10,000-turn cap limits scores to about 347.5k and makes 32768 effectively unreachable. Fitting 32768 into 10,000 turns would need ≥63.8% of spawns to be 4s, a ~27σ event.

## Quick start

```bash
conda env create -f environment.yml
conda activate rl_fun_times

python -m pytest          # engine parity, n-tuple, expectimax tests
python test_tournament.py # multiprocess tournament test
python benchmark_engine.py
```

**Pretrained model:** the 6-cell network (`ntuple_6cell_10shapes_1510193.npz`, 159 MB) is too large for git. Download it from this repo's **Releases** page into `models/ntuple_files/`. The small 4-cell networks used by the tests are included.

```bash
python rl_play.py        # batch evaluation: 500 games, prints best single game + replay seed
python human_play_rl.py  # watch the agent in the GUI, one move per keypress
python train_ntuple.py   # train from scratch or resume (Ctrl-C saves a checkpoint)
python tournament.py     # rank all checkpoints, then expectimax finals
```

### Reproduce the best game

```bash
python replay_game.py    # RUN_SEED=20260727, GAME_NUMBER=3900, depth-3 search
```

## Repository layout

| File | Purpose |
|---|---|
| `fast_engine.py` | Pure-Python bitboard engine (reference implementation) |
| `engine_jit.py` | Numba port of the engine on `uint64` boards |
| `ntuple.py` | N-tuple network, TD / temporal-coherence / multi-stage training loops |
| `expectimax.py` | Expectimax search with transposition table and adaptive depth |
| `train_ntuple.py` | Training driver: checkpointing, auto-resume, stage promotion |
| `tournament.py` | Parallel checkpoint tournament over shared-memory weights |
| `harvest.py` | Parallel seeded game search for high-scoring runs |
| `depth_sweep.py` | Search-depth vs. strength evaluation |
| `replay_game.py`, `game_seed.py` | Deterministic per-game seeding and GUI replay |
| `models/NTupleAgentInference.py` | Agent wrapper used by the play scripts (routes stages, runs search) |
| `test_*.py` | Parity, correctness, and determinism tests |
| `envs/`, `algos/`, other `models/`, `*_play*.py`, `rl_training.py` | Course-provided environment, baselines, and runners |

## References

- M. Szubert & W. Jaśkowski, *Temporal Difference Learning of N-Tuple Networks for the Game 2048*, IEEE CIG 2014.
- K.-H. Yeh, I.-C. Wu, C.-H. Hsueh, C.-C. Chang, C.-C. Liang, H. Chiang, *Multi-Stage Temporal Difference Learning for 2048-like Games*, IEEE TCIAIG 2016.
- W. Jaśkowski, *Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping*, IEEE TCIAIG 2017.

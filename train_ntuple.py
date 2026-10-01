"""Trains an n-tuple TD agent. Settable hyper parameters and file I/O.

Runnable:  python train_ntuple.py
Stop any time with Ctrl-C — progress is saved on exit and on a wall-clock
cadence. Checkpoints land in models/ntuple_files/; this run's newest
KEEP_CHECKPOINTS are kept (staged checkpoints are large). To continue a run,
set RESUME_FROM to a checkpoint path; resuming a single-stage checkpoint with
stages enabled promotes it (every stage starts as a copy of its tables).
"""

import glob
import os
import time
from collections import Counter

import numpy as np

import engine_jit as ej
import ntuple

# hyperparameters
# SHAPES = ntuple.SHAPES_4CELL
SHAPES = ntuple.SHAPES_6CELL
# stage boundaries as max-tile exponents: a new table set takes over when the
# board's largest tile reaches each threshold. () trains a single stage.
STAGE_THRESHOLDS = (13, 14)  # 8192 and 16384
N_EPISODES = 50_000_000  # a ceiling, not a target; stop by reading the curve
USE_TC = True  # per-weight adaptive rates; ALPHA is the base rate
ALPHA = 1.0 if USE_TC else 0.1
SEED = 1
PRINT_EVERY = 25_000
CHECKPOINT_MINUTES = 60
KEEP_CHECKPOINTS = 2  # newest N checkpoints of THIS run; older runs untouched
# "auto" continues this run's newest checkpoint, falling back to PROMOTE_FROM
# on a first launch; set to an explicit path or None to override
RESUME_FROM = "auto"
PROMOTE_FROM = os.path.join(
    "models", "ntuple_files", "ntuple_6cell_10shapes_3030992.npz"
)
OUT_DIR = os.path.join("models", "ntuple_files")

# these rates are the headline metric, not average score
TILE_MARKS = ((4096, 12), (8192, 13), (16384, 14))

if STAGE_THRESHOLDS and not USE_TC:
    raise SystemExit("staged training requires USE_TC = True")

os.makedirs(OUT_DIR, exist_ok=True)
shifts, bases, n_weights = ntuple.build_placements(SHAPES)
n_stages = len(STAGE_THRESHOLDS) + 1
thresholds_arr = np.array(STAGE_THRESHOLDS, dtype=np.int64)

run_tag = f"ntuple_{shifts.shape[1]}cell_{len(SHAPES)}shapes"
if USE_TC:
    run_tag += "_tc"
if n_stages > 1:
    run_tag += f"_ms{n_stages}"

if RESUME_FROM == "auto":
    # newest checkpoint of this exact run configuration, else the promotion
    # source; prevents a restart from silently discarding a run's progress
    existing = sorted(
        glob.glob(os.path.join(OUT_DIR, f"{run_tag}_[0-9]*.npz")),
        key=os.path.getmtime,
    )
    RESUME_FROM = existing[-1] if existing else PROMOTE_FROM
    if RESUME_FROM is not None and not os.path.exists(RESUME_FROM):
        RESUME_FROM = None

if RESUME_FROM is not None:
    weights, ckpt_shapes, games_done, ckpt_thresholds = (
        ntuple.load_checkpoint_full(RESUME_FROM)
    )
    if ckpt_shapes != tuple(SHAPES):
        raise ValueError("checkpoint shape set does not match SHAPES")
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.shape[0] == 1 and n_stages > 1:
        print(f"promoting single-stage checkpoint to {n_stages} stages "
              f"(each stage starts as a copy)")
        weights = np.repeat(weights, n_stages, axis=0)
    elif weights.shape[0] != n_stages:
        raise ValueError(
            f"checkpoint has {weights.shape[0]} stage(s); configured for "
            f"{n_stages}"
        )
    elif ckpt_thresholds is not None and tuple(ckpt_thresholds) != STAGE_THRESHOLDS:
        raise ValueError("checkpoint stage thresholds do not match config")
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    print(f"resuming from {RESUME_FROM} at {games_done:,} games")
else:
    weights = np.zeros((n_stages, n_weights), dtype=np.float32)
    games_done = 0
# offset the seed on resume so the continued run gets a fresh spawn stream
ej.seed_rng(SEED + games_done)

tc_state_path = os.path.join(OUT_DIR, f"{run_tag}_accumulators.npz")
if USE_TC:
    e_acc = a_acc = None
    if os.path.exists(tc_state_path):
        with np.load(tc_state_path) as data:
            if data["e_acc"].shape == weights.shape:
                e_acc = data["e_acc"]
                a_acc = data["a_acc"]
                print(f"loaded temporal-coherence accumulators from "
                      f"{tc_state_path}")
    if e_acc is None:
        e_acc = np.zeros_like(weights)
        a_acc = np.zeros_like(weights)
else:
    e_acc = a_acc = None

footprint = weights.nbytes * (3 if USE_TC else 1) / 2**30
print(f"{run_tag}: {shifts.shape[0]} placements, {n_stages} stage(s) x "
      f"{n_weights:,} weights ({footprint:.2f} GB resident), alpha={ALPHA}, "
      f"{'temporal coherence' if USE_TC else 'fixed rate'}, seed={SEED}, "
      f"up to {N_EPISODES:,} episodes, checkpoint every "
      f"{CHECKPOINT_MINUTES} min, keep newest {KEEP_CHECKPOINTS}")


def save_all(episode_count):
    path = os.path.join(OUT_DIR, f"{run_tag}_{episode_count}.npz")
    try:
        ntuple.save_checkpoint(
            path, weights, SHAPES, episode_count, ALPHA, [],
            stage_thresholds=STAGE_THRESHOLDS if n_stages > 1 else None,
        )
        if USE_TC:
            np.savez(tc_state_path, e_acc=e_acc, a_acc=a_acc)
    except OSError as err:
        # a failed write leaves a truncated file that would poison auto-resume
        # and any tool that picks the newest checkpoint
        if os.path.exists(path):
            os.remove(path)
        print(f"CHECKPOINT FAILED ({err}); removed partial file, run continues")
        return "<not saved>"
    # prune this run's oldest checkpoints; other runs' files never match the tag
    pattern = os.path.join(OUT_DIR, f"{run_tag}_[0-9]*.npz")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    for stale in files[:-KEEP_CHECKPOINTS]:
        os.remove(stale)
        print(f"pruned old checkpoint: {stale}")
    return path


window_scores = []
window_tiles = Counter()
window_decisions = 0
overall_tiles = Counter()
episode = games_done
window_start = time.perf_counter()
run_start = window_start
last_checkpoint = window_start

overflow_episodes = 0

try:
    while episode < N_EPISODES:
        episode += 1
        try:
            if n_stages > 1 or USE_TC:
                score, moves, max_exp = ntuple.play_training_episode_staged(
                    weights, e_acc, a_acc, thresholds_arr, shifts, bases, ALPHA
                )
            else:
                score, moves, max_exp = ntuple.play_training_episode(
                    weights[0], shifts, bases, ALPHA
                )
        except AssertionError as err:
            # engine refused a merge past 2**15; abandon this episode but
            # never the run. Reproducible: same seed + resume offset replays
            # the identical episode sequence
            overflow_episodes += 1
            print(f"episode {episode:,}: engine overflow ({err}) — episode "
                  f"abandoned, run continues [seed={SEED}, "
                  f"resume_offset={games_done}]")
            continue
        window_scores.append(score)
        window_tiles[max_exp] += 1
        overall_tiles[max_exp] += 1
        window_decisions += moves

        if episode % PRINT_EVERY == 0 and window_scores:
            n = len(window_scores)
            elapsed = time.perf_counter() - window_start
            rates = "  ".join(
                f"P(>={tile}) {100 * sum(c for e, c in window_tiles.items() if e >= exp) / n:.3f}%"
                for tile, exp in TILE_MARKS
            )
            print(f"episode {episode:>9,}: avg {sum(window_scores) / n:>8,.0f}  "
                  f"max {max(window_scores):>7,}  {rates}  "
                  f"{window_decisions / elapsed:,.0f} dec/s")
            window_scores = []
            window_tiles = Counter()
            window_decisions = 0
            window_start = time.perf_counter()

        if time.perf_counter() - last_checkpoint > CHECKPOINT_MINUTES * 60:
            print(f"checkpoint saved: {save_all(episode)}")
            last_checkpoint = time.perf_counter()
except KeyboardInterrupt:
    print("\ninterrupted — saving progress")

final_path = save_all(episode)
total_s = time.perf_counter() - run_start
print(f"\nstopped at {episode:,} games in {total_s / 3600:.2f} h; "
      f"saved {final_path}")
print(f"max-tile counts (this run): "
      f"{dict(sorted((1 << e, c) for e, c in overall_tiles.items()))}")
if overflow_episodes:
    print(f"overflow-abandoned episodes: {overflow_episodes}")
print(f"to continue: set RESUME_FROM to the saved path above")

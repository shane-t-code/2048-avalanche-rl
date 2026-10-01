"""Expectimax depth sweep: max-tile distribution per depth for a checkpoint.

Runnable:  python depth_sweep.py
The ranking metric is the max-tile distribution (P(>=tile) rates), not mean
score. Each game is exception-guarded so one refused overflow merge logs an
outlier instead of ending the sweep.
"""

import glob
import os
import time
from collections import Counter

import numpy as np

import engine_jit as ej
import ntuple
from expectimax import ExpectimaxSearch

# settings
# newest 6-cell checkpoint by default; replace with an explicit path to pin
CHECKPOINT = max(
    glob.glob(os.path.join("models", "ntuple_files", "ntuple_6cell_*.npz")),
    key=os.path.getmtime,
)
DEPTHS = (1, 2, 3)
N_GAMES = 100
SEED = 20260729
CHANCE_CAP = 6
TIME_BUDGET_MS = None
DEEPEN_EMPTY_THRESHOLD = 4  # extra ply when this few cells remain (0 = off)
DEEPEN_EXTRA = 1
TILE_MARKS = (2048, 4096, 8192, 16384)

weights, shapes, games_trained = ntuple.load_checkpoint(CHECKPOINT)
shifts, bases, _ = ntuple.build_placements(shapes)
print(f"checkpoint {CHECKPOINT} ({games_trained:,} games trained), "
      f"{N_GAMES} games per depth\n")

for depth in DEPTHS:
    search = ExpectimaxSearch(
        weights, shifts, bases, depth=depth, chance_cap=CHANCE_CAP,
        time_budget_ms=TIME_BUDGET_MS,
        deepen_empty_threshold=DEEPEN_EMPTY_THRESHOLD,
        deepen_extra=DEEPEN_EXTRA,
    )
    ej.seed_rng(SEED)
    tiles = Counter()
    scores = []
    moves_total = 0
    outliers = 0
    start = time.perf_counter()
    for game in range(N_GAMES):
        try:
            # jitted functions box uint64 returns as plain int; wrap before
            # passing back in, or boards with a large tile in the last cell
            # overflow the int64 conversion
            board = np.uint64(ej.spawn(np.uint64(ej.spawn(ej.ZERO))))
            score = 0
            while True:
                action = search.best_action(board)
                if action < 0:
                    break
                after, reward, _ = ej.slide(board, action)
                after = np.uint64(after)
                score += int(reward)
                moves_total += 1
                if ej.count_empty(after):
                    board = np.uint64(ej.spawn(after))
                else:
                    board = after
            tiles[int(ej.max_exponent(board))] += 1
            scores.append(score)
        except AssertionError as err:
            # merge past 2**15 refused: record and continue the sweep
            outliers += 1
            print(f"  depth {depth} game {game}: overflow outlier ({err})")
    elapsed = time.perf_counter() - start

    n = len(scores)
    dist = dict(sorted((1 << e, c) for e, c in tiles.items()))
    rates = "  ".join(
        f"P(>={tile}) {100 * sum(c for e, c in tiles.items() if (1 << e) >= tile) / n:.1f}%"
        for tile in TILE_MARKS
    )
    print(f"depth {depth}: {rates}")
    print(f"  max-tile counts: {dist}")
    print(f"  score avg {sum(scores) / n:,.0f}, max {max(scores):,}; "
          f"{moves_total / elapsed:,.0f} moves/sec, "
          f"{elapsed / n:.2f} s/game"
          + (f"; overflow outliers: {outliers}" if outliers else ""))
    print()

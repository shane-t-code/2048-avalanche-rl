"""Checkpoint tournament: rank all n-tuple checkpoints by fast greedy play,
then re-run the leaders under expectimax search.

Runnable:  python tournament.py
Final ranking is best highest-tile first, score as tiebreak, with the source
checkpoint recorded for each. Games fan out over a process pool; each
checkpoint's weight table is placed in shared memory once and mapped
read-only by every worker. Every game is exception-guarded: an overflow
refusal (merge past 2**15) logs its chunk seed + game index as an outlier
and never kills a worker. Chunk seeds are deterministic, so any game —
including outliers — can be replayed exactly.
"""

import glob
import os
import time
from collections import Counter
from multiprocessing import Pool, shared_memory

import numpy as np

import engine_jit as ej
import ntuple
from expectimax import ExpectimaxSearch

# settings
CHECKPOINT_DIR = os.path.join("models", "ntuple_files")
CHECKPOINT_GLOB = "ntuple_6cell_*.npz"
GREEDY_GAMES = 1_000
TOP_N = 6
SEARCH_GAMES = 5_000
SEARCH_DEPTH = 2
CHANCE_CAP = 6
TIME_BUDGET_MS = None
DEEPEN_EMPTY_THRESHOLD = 4
DEEPEN_EXTRA = 1
WORKERS = 6
BASE_SEED = 20260730

TILE_MARKS = (2048, 4096, 8192, 16384)


def _play_chunk(args):
    """Worker task: one chunk of games against a shared weight table."""
    (shm_name, n_weights, shapes, chunk_seed, n_games, mode, depth,
     chance_cap, time_budget_ms, deepen_threshold, deepen_extra) = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        weights = np.ndarray((n_weights,), dtype=np.float32, buffer=shm.buf)
        shifts, bases, expected = ntuple.build_placements(shapes)
        if expected != n_weights:
            raise ValueError("weight count does not match shape set")
        search = None
        if mode == "search":
            search = ExpectimaxSearch(
                weights, shifts, bases, depth=depth, chance_cap=chance_cap,
                time_budget_ms=time_budget_ms,
                deepen_empty_threshold=deepen_threshold,
                deepen_extra=deepen_extra,
            )
        ej.seed_rng(chunk_seed)
        scores = []
        tiles = []
        outliers = []
        for game in range(n_games):
            try:
                if mode == "greedy":
                    score, _, max_exp = ntuple.play_greedy_episode(
                        weights, shifts, bases
                    )
                else:
                    # jitted functions box uint64 returns as plain int; wrap
                    # before passing back in, or boards with a large tile in
                    # the last cell overflow the int64 conversion
                    board = np.uint64(ej.spawn(np.uint64(ej.spawn(ej.ZERO))))
                    score = 0
                    while True:
                        action = search.best_action(board)
                        if action < 0:
                            break
                        after, reward, _ = ej.slide(board, action)
                        after = np.uint64(after)
                        score += int(reward)
                        if ej.count_empty(after):
                            board = np.uint64(ej.spawn(after))
                        else:
                            board = after
                    max_exp = int(ej.max_exponent(board))
                scores.append(int(score))
                tiles.append(int(max_exp))
            except AssertionError as err:
                outliers.append((chunk_seed, game, str(err)))
        return {
            "chunk_seed": chunk_seed,
            "scores": scores,
            "tiles": tiles,
            "outliers": outliers,
        }
    finally:
        shm.close()


def _evaluate_checkpoint(pool, workers, path, n_games, mode, seed_block,
                         depth=1, chance_cap=CHANCE_CAP,
                         time_budget_ms=TIME_BUDGET_MS,
                         deepen_threshold=DEEPEN_EMPTY_THRESHOLD,
                         deepen_extra=DEEPEN_EXTRA):
    """Run n_games for one checkpoint across the pool; aggregate results."""
    weights, shapes, games_trained = ntuple.load_checkpoint(path)
    shm = shared_memory.SharedMemory(create=True, size=weights.nbytes)
    try:
        view = np.ndarray(weights.shape, dtype=np.float32, buffer=shm.buf)
        view[:] = weights
        n_weights = len(weights)
        del weights, view

        per_chunk = [n_games // workers] * workers
        for i in range(n_games % workers):
            per_chunk[i] += 1
        tasks = [
            (shm.name, n_weights, shapes, seed_block + i, chunk, mode, depth,
             chance_cap, time_budget_ms, deepen_threshold, deepen_extra)
            for i, chunk in enumerate(per_chunk) if chunk > 0
        ]
        results = pool.map(_play_chunk, tasks)
    finally:
        shm.close()
        shm.unlink()

    games = []  # (max_exp, score, chunk_seed, game_index)
    outliers = []
    for result in results:
        for i, (score, exp) in enumerate(zip(result["scores"], result["tiles"])):
            games.append((exp, score, result["chunk_seed"], i))
        outliers.extend(result["outliers"])
    if not games:
        raise RuntimeError(f"{path}: every game failed ({len(outliers)} outliers)")
    scores = [g[1] for g in games]
    tile_counts = Counter(g[0] for g in games)
    best_exp, best_score, best_seed, best_index = max(games)
    return {
        "path": path,
        "games_trained": games_trained,
        "n_games": len(games),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "best_tile": 1 << best_exp,
        "best_game": (best_seed, best_index, 1 << best_exp, best_score),
        "tile_counts": tile_counts,
        "outliers": outliers,
    }


def _tile_rates(summary):
    n = summary["n_games"]
    return "  ".join(
        f"P(>={tile}) "
        f"{100 * sum(c for e, c in summary['tile_counts'].items() if (1 << e) >= tile) / n:.2f}%"
        for tile in TILE_MARKS
    )


def _report_outliers(summary):
    for chunk_seed, game_index, message in summary["outliers"]:
        print(f"  OUTLIER {os.path.basename(summary['path'])}: "
              f"chunk_seed={chunk_seed} game={game_index} ({message})")


def rank_checkpoints(paths, pool, workers, n_games, base_seed):
    """Phase 1: rank every checkpoint by average score over fast greedy games."""
    summaries = []
    for index, path in enumerate(paths):
        summary = _evaluate_checkpoint(
            pool, workers, path, n_games, "greedy",
            seed_block=base_seed + index * 10_000,
        )
        summaries.append(summary)
        print(f"ranked {os.path.basename(path)}: "
              f"avg {summary['avg_score']:>9,.0f}  "
              f"best tile {summary['best_tile']:>5}  ({summary['n_games']} games)")
        _report_outliers(summary)
    summaries.sort(key=lambda s: s["avg_score"], reverse=True)
    return summaries


def search_checkpoints(summaries, pool, workers, n_games, depth, base_seed):
    """Phase 2: re-run the leaders under expectimax; rank by tile then score."""
    results = []
    for index, ranked in enumerate(summaries):
        summary = _evaluate_checkpoint(
            pool, workers, ranked["path"], n_games, "search",
            seed_block=base_seed + 1_000_000 + index * 10_000, depth=depth,
        )
        results.append(summary)
        print(f"searched {os.path.basename(summary['path'])} (depth {depth}): "
              f"best tile {summary['best_tile']:>5}  "
              f"best score {summary['max_score']:>8,}  {_tile_rates(summary)}")
        _report_outliers(summary)
    results.sort(key=lambda s: (s["best_tile"], s["max_score"]), reverse=True)
    return results


def main():
    paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, CHECKPOINT_GLOB)))
    if not paths:
        raise SystemExit(f"no checkpoints match {CHECKPOINT_GLOB}")
    print(f"{len(paths)} checkpoints, {WORKERS} workers; greedy ranking over "
          f"{GREEDY_GAMES:,} games each\n")
    start = time.perf_counter()
    with Pool(WORKERS) as pool:
        ranked = rank_checkpoints(paths, pool, WORKERS, GREEDY_GAMES, BASE_SEED)
        finalists = ranked[:TOP_N]
        print(f"\ntop {len(finalists)} -> expectimax depth {SEARCH_DEPTH}, "
              f"{SEARCH_GAMES:,} games each\n")
        final = search_checkpoints(
            finalists, pool, WORKERS, SEARCH_GAMES, SEARCH_DEPTH, BASE_SEED
        )

    print("\n=== final ranking (best tile, then best score) ===")
    for place, summary in enumerate(final, 1):
        print(f"{place:>2}. {os.path.basename(summary['path'])}: "
              f"tile {summary['best_tile']}, score {summary['max_score']:,}, "
              f"avg {summary['avg_score']:,.0f}")
    best = final[0]
    seed, game_index, tile, score = best["best_game"]
    print(f"\nbest highest-tile game: {tile} (score {score:,}) from "
          f"{best['path']} [chunk_seed={seed} game={game_index}]")
    top_score = max(final, key=lambda s: s["max_score"])
    print(f"best score game: {top_score['max_score']:,} from {top_score['path']}")
    print(f"\ntotal {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

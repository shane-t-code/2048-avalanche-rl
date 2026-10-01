"""Tests for tournament.py using the small 4-cell checkpoints only.

Runnable directly (no pytest needed):  python test_tournament.py

Uses tiny game counts and 2 workers so it stays quick and can run alongside
a training run.
"""

import glob
import os
import time
from multiprocessing import Pool

import tournament

SMALL_GLOB = os.path.join("models", "ntuple_files", "ntuple_4cell_2shapes_*.npz")
SEED = 20260731
WORKERS = 2


def _small_paths():
    paths = sorted(glob.glob(SMALL_GLOB))
    assert len(paths) >= 3, f"need 4-cell checkpoints, found {len(paths)}"
    return paths


def check_ranking(pool, paths):
    ranked = tournament.rank_checkpoints(paths, pool, WORKERS, 40, SEED)
    assert len(ranked) == len(paths)
    assert all(s["n_games"] == 40 for s in ranked)
    assert all(s["avg_score"] > 0 for s in ranked)
    assert all(not s["outliers"] for s in ranked)
    averages = [s["avg_score"] for s in ranked]
    assert averages == sorted(averages, reverse=True), "not sorted by avg"
    by_path = {s["path"]: s["avg_score"] for s in ranked}
    earliest = by_path[os.path.join("models", "ntuple_files",
                                    "ntuple_4cell_2shapes_25000.npz")]
    final = by_path[os.path.join("models", "ntuple_files",
                                 "ntuple_4cell_2shapes_final.npz")]
    assert final > earliest * 1.1, (
        f"final checkpoint ({final:.0f}) should clearly beat the earliest "
        f"({earliest:.0f})"
    )
    print(f"PASS ranking ({len(paths)} checkpoints x 40 greedy games, "
          f"sorted, final beats earliest)")
    return ranked


def check_ranking_deterministic(pool, paths, first_run):
    again = tournament.rank_checkpoints(paths[:3], pool, WORKERS, 40, SEED)
    reference = {s["path"]: s for s in first_run}
    for summary in again:
        ref = reference[summary["path"]]
        assert summary["avg_score"] == ref["avg_score"]
        assert summary["best_game"] == ref["best_game"]
        assert summary["tile_counts"] == ref["tile_counts"]
    print("PASS ranking determinism (same seeds -> identical results)")


def check_search_phase(pool, ranked):
    final = tournament.search_checkpoints(ranked[:2], pool, WORKERS, 8, 2, SEED)
    assert len(final) == 2
    for summary in final:
        assert summary["n_games"] == 8
        assert summary["best_tile"] >= 256
        assert summary["max_score"] > 0
        seed, game_index, tile, score = summary["best_game"]
        assert tile == summary["best_tile"]
        assert not summary["outliers"]
    keys = [(s["best_tile"], s["max_score"]) for s in final]
    assert keys == sorted(keys, reverse=True), "not sorted by tile then score"
    print("PASS search phase (top 2 x 8 depth-2 games, attribution + order)")


def main():
    start = time.perf_counter()
    paths = _small_paths()
    with Pool(WORKERS) as pool:
        ranked = check_ranking(pool, paths)
        check_ranking_deterministic(pool, paths, ranked)
        check_search_phase(pool, ranked)
    print(f"\nALL TESTS PASSED in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()

"""Searches for a high-scoring game across a process pool.

Runnable:  python harvest.py

Games use the same per-game seeding as rl_play.py, so any game found here is
reproducible: replay it in the GUI with replay_game.py, or reach it in
rl_play.py by using the same run_seed and a large enough num_games.
"""

import glob
import os
import signal
import time
from collections import Counter
from multiprocessing import Pool

from envs.SGAI_2048 import SGAI_2048
from game_seed import seed_game
from models.NTupleAgentInference import NTupleAgentInference

# settings
RUN_SEED = 20260727
START_GAME = 104_000
N_GAMES = 40_000
WORKERS = 6
SEARCH_DEPTH = 3
TARGET_TILE = 16384
TARGET_SCORE = 310_000  # report games at or above this
MAX_TURNS = 50_000  # safety valve only; games end on their own well before this
# "auto" picks the newest training checkpoint matching MODEL_GLOB, ignoring
# any file written in the last two minutes (a concurrent trainer may still be
# writing it). Resolved once in the parent so every worker shares one model.
MODEL_PATH = "auto"
MODEL_GLOB = os.path.join("models", "ntuple_files", "ntuple_6cell_*.npz")

_agent = None
_env = None


def resolve_model_path():
    if MODEL_PATH != "auto":
        return MODEL_PATH
    cutoff = time.time() - 120
    candidates = [
        path for path in glob.glob(MODEL_GLOB)
        if "accumulators" not in path and os.path.getmtime(path) < cutoff
    ]
    if not candidates:
        raise SystemExit(f"no settled checkpoint matches {MODEL_GLOB}")
    return max(candidates, key=os.path.getmtime)


def _init_worker(model_path):
    global _agent, _env
    # the parent owns Ctrl-C; without this each worker raises its own
    # KeyboardInterrupt and the pool respawns replacements that outlive it
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _agent = NTupleAgentInference(
        model_file_path=model_path,
        search_depth=SEARCH_DEPTH,
        deepen_empty_threshold=4,
        deepen_extra=1,
    )
    _env = SGAI_2048(render_mode=None)


def _play(game_number):
    try:
        seed_game(RUN_SEED, game_number)
        obs_, _ = _env.reset()
        for _ in range(MAX_TURNS):
            action = _agent.get_action(obs_raw=obs_)
            obs_, reward, terminated, truncated, _ = _env.step(action)
            if terminated or truncated:
                break
        return game_number, int(_env.highest_tile), int(_env.total_score), None
    except AssertionError as err:
        # engine refused a merge past 2**15; drop this game, keep the search
        return game_number, 0, 0, str(err)


def main():
    model_path = resolve_model_path()
    print(f"model: {model_path}")
    print(f"harvesting games {START_GAME:,}-{START_GAME + N_GAMES - 1:,} "
          f"(run_seed={RUN_SEED}) on {WORKERS} workers at depth "
          f"{SEARCH_DEPTH}")
    print(f"reporting games reaching tile {TARGET_TILE} with score "
          f">= {TARGET_SCORE:,}\n")
    tiles = Counter()
    hits = []
    best = (0, 0, -1)  # tile, score, game number
    done = 0
    start = time.perf_counter()
    pool = Pool(WORKERS, initializer=_init_worker, initargs=(model_path,))
    try:
        for game_number, tile, score, error in pool.imap_unordered(
            _play, range(START_GAME, START_GAME + N_GAMES), chunksize=4
        ):
            done += 1
            if error is not None:
                print(f"  OUTLIER game {game_number}: {error}")
                continue
            tiles[tile] += 1
            if (tile, score) > (best[0], best[1]):
                best = (tile, score, game_number)
            if tile >= TARGET_TILE and score >= TARGET_SCORE:
                hits.append((game_number, tile, score))
                print(f"  HIT game {game_number}: tile {tile}, "
                      f"score {score:,}")
            if done % 100 == 0:
                rate = done / (time.perf_counter() - start)
                remaining = (N_GAMES - done) / rate / 60
                big = sum(c for t, c in tiles.items() if t >= TARGET_TILE)
                print(f"  {done:,}/{N_GAMES:,} games, {big} at "
                      f"{TARGET_TILE}, best {best[0]}/{best[1]:,} "
                      f"(game {best[2]}), ~{remaining:.0f} min left")
        pool.close()
    except KeyboardInterrupt:
        print("\ninterrupted — stopping workers; results below cover "
              f"{done:,} completed games")
        pool.terminate()
    finally:
        pool.join()

    elapsed = (time.perf_counter() - start) / 60
    print(f"\ndone in {elapsed:.0f} min")
    print(f"max-tile counts: {dict(sorted(tiles.items()))}")
    tile, score, game_number = best
    print(f"best game: #{game_number} - tile {tile}, score {score:,}")
    if hits:
        hits.sort(key=lambda h: h[2], reverse=True)
        print(f"\n{len(hits)} game(s) met the target; best is game "
              f"{hits[0][0]} at {hits[0][2]:,}")
    print(f"\nreplay any of these: set RUN_SEED={RUN_SEED} and "
          f"GAME_NUMBER=<number> in replay_game.py")


if __name__ == "__main__":
    main()

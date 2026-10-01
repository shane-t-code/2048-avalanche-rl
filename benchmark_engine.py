"""Throughput benchmark for fast_engine.py.

Runnable directly:  python benchmark_engine.py

Reports single-core rates for raw slides, a training-loop-shaped workload
(all afterstates computed once per turn and reused), and the class env's
step() for comparison.
"""

import random
import time

_t0 = time.perf_counter()
import fast_engine as fe  # noqa: E402  (timed import: includes table build)

_TABLE_BUILD_S = time.perf_counter() - _t0

SEED = 20260725


def _playout_boards(n_boards, rng):
    """Collect boards from random-legal-move games for a realistic mix."""
    boards = []
    board = fe.spawn(fe.spawn(0, rng), rng)
    while len(boards) < n_boards:
        moves = fe.afterstates(board)
        if not moves:
            board = fe.spawn(fe.spawn(0, rng), rng)
            continue
        _, board, _ = rng.choice(moves)
        if fe.count_empty(board):
            board = fe.spawn(board, rng)
        boards.append(board)
    return boards


def bench_raw_slides(n=200_000):
    rng = random.Random(SEED)
    boards = _playout_boards(20_000, rng)
    n_boards = len(boards)
    slide = fe.slide
    start = time.perf_counter()
    for i in range(n):
        slide(boards[i % n_boards], i & 3)
    elapsed = time.perf_counter() - start
    return n / elapsed


def bench_training_loop_shape(n_games=2_000):
    """Full games: one afterstates() call per turn, results reused for the
    action choice and legality, then a spawn. Mirrors how the TD trainer
    will drive the engine (minus value lookups)."""
    rng = random.Random(SEED + 1)
    decisions = 0
    slides = 0
    start = time.perf_counter()
    for _ in range(n_games):
        board = fe.spawn(fe.spawn(0, rng), rng)
        while True:
            moves = fe.afterstates(board)
            if not moves:
                break
            _, board, _ = rng.choice(moves)
            decisions += 1
            slides += 4
            if fe.count_empty(board):
                board = fe.spawn(board, rng)
    elapsed = time.perf_counter() - start
    return (
        decisions / elapsed,
        slides / elapsed,
        n_games / elapsed,
        decisions / n_games,
    )


def bench_env_steps(n_steps=3_000):
    from envs.SGAI_2048 import SGAI_2048

    rng = random.Random(SEED + 2)
    env = SGAI_2048()
    env.reset()
    start = time.perf_counter()
    for _ in range(n_steps):
        _, _, terminated, _, _ = env.step(rng.randrange(4))
        if terminated:
            env.reset()
    elapsed = time.perf_counter() - start
    return n_steps / elapsed


def main():
    print(f"table build (import time): {_TABLE_BUILD_S:.2f}s")

    rate = bench_raw_slides()
    print(f"raw slides:               {rate:,.0f} slides/sec")

    decisions, slides, games, avg_turns = bench_training_loop_shape()
    print(f"training-loop shape:      {decisions:,.0f} decisions/sec "
          f"({slides:,.0f} slides/sec, {games:,.1f} games/sec, "
          f"avg {avg_turns:.0f} turns/game)")

    env_rate = bench_env_steps()
    print(f"class env step():         {env_rate:,.0f} steps/sec "
          f"(engine decision rate is {decisions / env_rate:,.0f}x)")


if __name__ == "__main__":
    main()

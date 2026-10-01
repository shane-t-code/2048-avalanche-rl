"""Parity tests for engine_jit.py against fast_engine.py.

Runnable directly (no pytest needed):  python test_engine_jit.py

Includes a per-cell high-tile sweep because signed-shift sign extension only
bites when specific cells hold large exponents.
"""

import random
import time

import numpy as np

import engine_jit as ej
import fast_engine as fe

SEED = 20260726
U64 = np.uint64


def _check_board(board):
    """Compare jit vs reference on one board, all four actions."""
    for action in fe.ACTIONS:
        try:
            ref = fe.slide(board, action)
            ref_raised = False
        except AssertionError:
            ref_raised = True
        try:
            nb, s, changed = ej.slide(U64(board), action)
            jit_raised = False
        except AssertionError:
            jit_raised = True
        assert ref_raised == jit_raised, (
            f"overflow behavior mismatch: ref_raised={ref_raised} "
            f"jit_raised={jit_raised} action={action} board={board:#018x}"
        )
        if ref_raised:
            continue
        ref_after, ref_score, ref_changed = ref
        assert int(nb) == ref_after and int(s) == ref_score, (
            f"slide mismatch action={action} board={board:#018x}"
        )
        assert bool(changed) == ref_changed


def test_playout_board_parity(n_boards=50_000):
    rng = random.Random(SEED)
    board = fe.spawn(fe.spawn(0, rng), rng)
    checked = 0
    while checked < n_boards:
        _check_board(board)
        checked += 1
        moves = fe.afterstates(board)
        if not moves:
            board = fe.spawn(fe.spawn(0, rng), rng)
            continue
        _, board, _ = rng.choice(moves)
        if fe.count_empty(board):
            board = fe.spawn(board, rng)
    print(f"PASS jit slide parity on playout boards ({checked:,} boards x 4 actions)")


def test_high_tile_cell_sweep(boards_per_case=60):
    """Force a large exponent into every individual cell position."""
    rng = random.Random(SEED + 1)
    checked = 0
    for cell in range(16):
        for exp in range(8, 16):
            for _ in range(boards_per_case):
                board = exp << (4 * cell)
                for other in range(16):
                    if other != cell and rng.random() < 0.5:
                        board |= rng.randint(1, 14) << (4 * other)
                _check_board(board)
                checked += 1
    print(f"PASS high-tile cell sweep ({checked:,} boards x 4 actions, "
          f"exponents 8-15 in every cell)")


def test_afterstates_into():
    rng = random.Random(SEED + 2)
    boards4 = np.empty(4, dtype=np.uint64)
    scores4 = np.empty(4, dtype=np.int64)
    actions4 = np.empty(4, dtype=np.int64)
    board = fe.spawn(fe.spawn(0, rng), rng)
    for _ in range(20_000):
        ref = fe.afterstates(board)
        n = ej.afterstates_into(U64(board), boards4, scores4, actions4)
        got = [
            (int(actions4[i]), int(boards4[i]), int(scores4[i])) for i in range(n)
        ]
        assert got == ref, f"afterstates mismatch on board={board:#018x}"
        assert ej.count_empty(U64(board)) == fe.count_empty(board)
        if not ref:
            board = fe.spawn(fe.spawn(0, rng), rng)
            continue
        _, board, _ = rng.choice(ref)
        if fe.count_empty(board):
            board = fe.spawn(board, rng)
    print("PASS afterstates_into + count_empty parity (20,000 boards)")


def test_jit_spawn_distribution(n=20_000):
    ej.seed_rng(SEED + 3)
    array = np.zeros((4, 4), dtype=int)
    array[0] = [2, 4, 2, 4]
    board = U64(fe.board_from_array(array))
    empty_cells = [i for i in range(16) if (int(board) >> (4 * i)) & 0xF == 0]
    value_counts = {1: 0, 2: 0}
    cell_counts = dict.fromkeys(empty_cells, 0)
    for _ in range(n):
        spawned = int(ej.spawn(board))
        diff = spawned ^ int(board)
        assert diff != 0 and spawned & int(board) == int(board)
        cell = (diff.bit_length() - 1) // 4
        exp = (spawned >> (4 * cell)) & 0xF
        assert cell in cell_counts and exp in (1, 2)
        value_counts[exp] += 1
        cell_counts[cell] += 1
    two_rate = value_counts[1] / n
    assert 0.46 < two_rate < 0.54, f"2-vs-4 rate off: {two_rate:.3f}"
    for cell, count in cell_counts.items():
        rate = count / n
        assert 0.065 < rate < 0.10, f"cell {cell} spawn rate off: {rate:.3f}"
    print(f"PASS jit spawn distribution ({n:,} spawns, 2-rate {two_rate:.3f})")


def main():
    start = time.perf_counter()
    test_playout_board_parity()
    test_high_tile_cell_sweep()
    test_afterstates_into()
    test_jit_spawn_distribution()
    elapsed = time.perf_counter() - start
    print(f"\nALL TESTS PASSED in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

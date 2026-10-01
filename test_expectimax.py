"""Tests for expectimax.py.

Runnable directly (no pytest needed):  python test_expectimax.py

Uses only the small 4-cell network so tables stay cache-resident.
"""

import os
import random
import time

import numpy as np

import engine_jit as ej
import expectimax as ex
import fast_engine as fe
import ntuple

SEED = 20260728
U64 = np.uint64
SMALL_CKPT = os.path.join(
    "models", "ntuple_files", "ntuple_4cell_2shapes_final.npz"
)


def _search(weights, shifts, bases, **kwargs):
    return ex.ExpectimaxSearch(weights, shifts, bases, **kwargs)


def _playout_boards(n, rng):
    boards = []
    board = fe.spawn(fe.spawn(0, rng), rng)
    while len(boards) < n:
        moves = fe.afterstates(board)
        if not moves:
            board = fe.spawn(fe.spawn(0, rng), rng)
            continue
        boards.append(board)
        _, board, _ = rng.choice(moves)
        if fe.count_empty(board):
            board = fe.spawn(board, rng)
    return boards


def _dense_boards(n, rng, max_empty):
    boards = []
    while len(boards) < n:
        board = 0
        cells = rng.sample(range(16), 16 - rng.randint(1, max_empty))
        for cell in cells:
            board |= rng.randint(1, 8) << (4 * cell)
        if fe.legal_actions(board) and fe.count_empty(board) <= max_empty:
            boards.append(board)
    return boards


def _ref_state_value(board, depth, weights, shifts, bases, cap):
    moves = fe.afterstates(board)
    if not moves:
        return 0.0
    best = -1e300
    for _, after, reward in moves:
        if depth <= 1:
            val = reward + float(
                ntuple.evaluate(U64(after), weights, shifts, bases)
            )
        else:
            val = reward + _ref_chance_value(
                after, depth, weights, shifts, bases, cap
            )
        if val > best:
            best = val
    return best


def _ref_chance_value(after, depth, weights, shifts, bases, cap):
    empties = [i for i in range(16) if (after >> (4 * i)) & 0xF == 0]
    m = len(empties)
    n_expand = min(m, cap)
    total = 0.0
    for j in range(n_expand):
        cell = empties[(j * m) // n_expand]  # same stride rule as the search
        for exp in (1, 2):
            total += _ref_state_value(
                after | (exp << (4 * cell)), depth - 1, weights, shifts,
                bases, cap,
            )
    return total / (2.0 * n_expand)


def _ref_best_action(board, depth, weights, shifts, bases, cap):
    moves = fe.afterstates(board)
    if not moves:
        return -1
    best_action = moves[0][0]
    best = -1e300
    for action, after, reward in moves:
        if depth <= 1:
            val = reward + float(
                ntuple.evaluate(U64(after), weights, shifts, bases)
            )
        else:
            val = reward + _ref_chance_value(
                after, depth, weights, shifts, bases, cap
            )
        if val > best:
            best = val
            best_action = action
    return best_action


def _weight_sets():
    rng = np.random.default_rng(SEED)
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)
    random_weights = rng.standard_normal(n_weights).astype(np.float32)
    trained_weights, shapes, _ = ntuple.load_checkpoint(SMALL_CKPT)
    assert shapes == ntuple.SHAPES_4CELL
    return shifts, bases, [("random", random_weights), ("trained", trained_weights)]


def test_depth1_equals_greedy(n_boards=3_000):
    rng = random.Random(SEED + 1)
    shifts, bases, weight_sets = _weight_sets()
    boards = _playout_boards(n_boards, rng)
    for label, weights in weight_sets:
        search = _search(weights, shifts, bases, depth=1)
        for board in boards:
            greedy = int(ntuple.choose_action(U64(board), weights, shifts, bases))
            got = search.best_action(board)
            assert got == greedy, (
                f"depth-1 disagrees with greedy ({label}): {got} vs {greedy} "
                f"on {board:#018x}"
            )
    print(f"PASS depth-1 == greedy choose_action ({n_boards:,} boards x "
          f"random + trained weights)")


def test_matches_reference(n_depth2=120, n_depth3=12):
    rng = random.Random(SEED + 2)
    shifts, bases, weight_sets = _weight_sets()
    # cap 16 = exhaustive expansion; cap 3 verifies the stride-sampling rule
    for depth, n_boards in ((2, n_depth2), (3, n_depth3)):
        boards = _dense_boards(n_boards, rng, max_empty=6)
        for label, weights in weight_sets:
            for cap in (16, 3):
                for tt_log2 in (0, 16):
                    search = _search(
                        weights, shifts, bases, depth=depth, chance_cap=cap,
                        tt_log2=tt_log2,
                    )
                    for board in boards:
                        got = search.best_action(board)
                        want = _ref_best_action(
                            board, depth, weights, shifts, bases, cap
                        )
                        assert got == want, (
                            f"depth-{depth} action mismatch ({label}, cap={cap}, "
                            f"tt_log2={tt_log2}): {got} vs {want} on {board:#018x}"
                        )
                        v_jit = ex._state_value(
                            U64(board), depth, weights, shifts, bases, cap,
                            search.tt_keys, search.tt_depths, search.tt_gens,
                            search.tt_vals, search.tt_shift, search.tt_mask,
                            np.uint32(search.gen + 1_000_000),
                        )
                        v_ref = _ref_state_value(
                            board, depth, weights, shifts, bases, cap
                        )
                        assert abs(v_jit - v_ref) < 1e-6 * max(1.0, abs(v_ref)), (
                            f"depth-{depth} value mismatch ({label}, cap={cap}): "
                            f"{v_jit} vs {v_ref}"
                        )
    print(f"PASS search matches independent reference (depth 2 x {n_depth2} "
          f"boards, depth 3 x {n_depth3}, caps 16+3, tt on+off, values and "
          f"actions)")


def test_tt_invariance(n_depth2=800, n_depth3=60):
    rng = random.Random(SEED + 3)
    shifts, bases, weight_sets = _weight_sets()
    label, weights = weight_sets[1]
    for depth, n_boards in ((2, n_depth2), (3, n_depth3)):
        boards = _playout_boards(n_boards, rng)
        tt_on = _search(weights, shifts, bases, depth=depth, tt_log2=20)
        tt_off = _search(weights, shifts, bases, depth=depth, tt_log2=0)
        for board in boards:
            a_on = tt_on.best_action(board)
            a_off = tt_off.best_action(board)
            assert a_on == a_off, (
                f"transposition table changed depth-{depth} action: "
                f"{a_on} vs {a_off} on {board:#018x}"
            )
    print(f"PASS transposition table invariance (depth 2 x {n_depth2}, "
          f"depth 3 x {n_depth3} open boards)")


def test_chance_cap(n_boards=300):
    rng = random.Random(SEED + 4)
    shifts, bases, weight_sets = _weight_sets()
    _, weights = weight_sets[1]
    # capped search on open boards: must stay legal and deterministic
    boards = _playout_boards(n_boards, rng)
    capped = _search(weights, shifts, bases, depth=2, chance_cap=3)
    for board in boards:
        action = capped.best_action(board)
        assert action in fe.legal_actions(board)
        assert action == capped.best_action(board), "cap made search nondeterministic"
    print(f"PASS chance-node cap (legal + deterministic on {n_boards} open "
          f"boards; stride math reference-checked above)")


def test_time_budget_and_adaptive_depth(n_boards=100):
    rng = random.Random(SEED + 5)
    shifts, bases, weight_sets = _weight_sets()
    _, weights = weight_sets[1]
    boards = _playout_boards(n_boards, rng)
    # a zero budget must still return the depth-1 (greedy) action
    tight = _search(weights, shifts, bases, depth=3, time_budget_ms=0.0)
    for board in boards:
        assert tight.best_action(board) == int(
            ntuple.choose_action(U64(board), weights, shifts, bases)
        )
    # always-on adaptive deepening must equal a plain deeper search
    adaptive = _search(
        weights, shifts, bases, depth=1, deepen_empty_threshold=16,
        deepen_extra=1,
    )
    plain2 = _search(weights, shifts, bases, depth=2)
    for board in boards[:50]:
        assert adaptive.best_action(board) == plain2.best_action(board)
    print(f"PASS time budget (zero budget -> greedy) + adaptive deepening "
          f"(== plain depth-2) on {n_boards} boards")


def test_terminal_and_env_integration():
    from envs.SGAI_2048 import SGAI_2048
    from models.NTupleAgentInference import NTupleAgentInference

    shifts, bases, weight_sets = _weight_sets()
    _, weights = weight_sets[1]
    terminal = fe.board_from_array(
        [[2, 4, 8, 16], [16, 8, 4, 2], [2, 4, 8, 16], [16, 8, 4, 2]]
    )
    assert fe.is_terminal(terminal)
    search = _search(weights, shifts, bases, depth=2)
    assert search.best_action(terminal) == -1

    agent = NTupleAgentInference(model_file_path=SMALL_CKPT, search_depth=2)
    env = SGAI_2048()
    for _ in range(2):
        obs, _ = env.reset()
        for _ in range(10_000):
            action = agent.get_action(obs_raw=obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            assert reward >= 0, "search agent played an illegal move"
            if terminated:
                break
        assert terminated
    print("PASS terminal handling + depth-2 agent in the real env "
          "(2 games, no illegal moves)")


def main():
    start = time.perf_counter()
    test_depth1_equals_greedy()
    test_matches_reference()
    test_tt_invariance()
    test_chance_cap()
    test_time_budget_and_adaptive_depth()
    test_terminal_and_env_integration()
    elapsed = time.perf_counter() - start
    print(f"\nALL TESTS PASSED in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

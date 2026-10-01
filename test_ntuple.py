"""Tests for ntuple.py: placement expansion, value evaluation, update
behavior, determinism, and that training actually learns.

Runnable directly (no pytest needed):  python test_ntuple.py
"""

import time

import numpy as np

import engine_jit as ej
import fast_engine as fe
import ntuple

SEED = 20260727
U64 = np.uint64


def test_placement_expansion():
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)
    assert shifts.shape == (16, 4)
    assert n_weights == 2 * 16**4
    cell_sets = [frozenset(int(s) // 4 for s in row) for row in shifts]
    row_sets = set(cell_sets[:8])
    square_sets = set(cell_sets[8:])
    assert row_sets == {
        frozenset({0, 1, 2, 3}),
        frozenset({12, 13, 14, 15}),
        frozenset({0, 4, 8, 12}),
        frozenset({3, 7, 11, 15}),
    }, "row-line placements are not the 4 edge lines"
    assert square_sets == {
        frozenset({0, 1, 4, 5}),
        frozenset({2, 3, 6, 7}),
        frozenset({8, 9, 12, 13}),
        frozenset({10, 11, 14, 15}),
    }, "2x2 placements are not the 4 corner squares"
    assert set(int(b) for b in bases[:8]) == {0}
    assert set(int(b) for b in bases[8:]) == {16**4}
    print("PASS placement expansion (8 placements per shape, correct cells)")


def test_6cell_shapes():
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_6CELL)
    n_shapes = len(ntuple.SHAPES_6CELL)
    assert shifts.shape == (n_shapes * 8, 6)
    assert n_weights == n_shapes * 16**6
    # placements of different base shapes must never coincide: a shape that is
    # a rotation/mirror of another would silently duplicate its table
    for shape_a in range(n_shapes):
        sets_a = {
            frozenset(int(s) // 4 for s in shifts[shape_a * 8 + i])
            for i in range(8)
        }
        for shape_b in range(shape_a + 1, n_shapes):
            sets_b = {
                frozenset(int(s) // 4 for s in shifts[shape_b * 8 + i])
                for i in range(8)
            }
            overlap = sets_a & sets_b
            assert not overlap, (
                f"shapes {shape_a} and {shape_b} share placements {overlap} — "
                f"same symmetry orbit"
            )
    for row in shifts:
        cells = [int(s) // 4 for s in row]
        assert len(set(cells)) == 6 and all(0 <= c <= 15 for c in cells)
    print(f"PASS 6-cell shapes ({n_shapes} shapes, all from distinct "
          f"symmetry orbits, {n_weights:,} weights = "
          f"{n_weights * 4 / 2**30:.2f} GB)")


def test_choose_action():
    import random

    rng = np.random.default_rng(SEED + 7)
    py_rng = random.Random(SEED + 7)
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)
    weights = rng.standard_normal(n_weights).astype(np.float32)
    board = fe.spawn(fe.spawn(0, py_rng), py_rng)
    checked = 0
    while checked < 2_000:
        moves = fe.afterstates(board)
        got = int(ntuple.choose_action(U64(board), weights, shifts, bases))
        if not moves:
            assert got == -1, "choose_action must return -1 on terminal boards"
            board = fe.spawn(fe.spawn(0, py_rng), py_rng)
            continue
        expected = max(
            moves,
            key=lambda m: m[2] + ntuple.evaluate(U64(m[1]), weights, shifts, bases),
        )[0]
        assert got == expected, f"choose_action {got} != reference {expected}"
        assert got in [m[0] for m in moves], "chose an illegal action"
        checked += 1
        _, board, _ = py_rng.choice(moves)
        if fe.count_empty(board):
            board = fe.spawn(board, py_rng)
    print("PASS choose_action (greedy, always legal, -1 at terminal, "
          "2,000 boards)")


def test_inference_agent():
    import os

    from envs.SGAI_2048 import SGAI_2048
    from models.NTupleAgentInference import NTupleAgentInference

    path = os.path.join(
        "models", "ntuple_files", "ntuple_4cell_2shapes_final.npz"
    )
    agent = NTupleAgentInference(model_file_path=path)
    env = SGAI_2048()
    for _ in range(3):
        obs, _ = env.reset()
        for _ in range(10_000):
            action = agent.get_action(obs_raw=obs)
            board_before = obs.copy()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert reward >= 0, "inference agent played an illegal move"
            assert not np.array_equal(board_before, obs)
            if terminated:
                break
        assert terminated, "game did not finish"
        assert env.total_score > 1_000, f"suspiciously low: {env.total_score}"
    print(f"PASS inference agent (3 env games, no illegal moves, trained on "
          f"{agent.games_trained:,} games)")


def _reference_v(board, weights, shifts, bases):
    total = 0.0
    for p in range(shifts.shape[0]):
        idx = 0
        for j in range(shifts.shape[1]):
            idx |= ((int(board) >> int(shifts[p][j])) & 0xF) << (4 * j)
        total += float(weights[int(bases[p]) + idx])
    return total


def test_value_evaluation():
    rng = np.random.default_rng(SEED)
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)
    weights = rng.standard_normal(n_weights).astype(np.float32)
    import random

    py_rng = random.Random(SEED)
    board = fe.spawn(fe.spawn(0, py_rng), py_rng)
    for _ in range(2_000):
        v_jit = ntuple.evaluate(U64(board), weights, shifts, bases)
        v_ref = _reference_v(board, weights, shifts, bases)
        assert abs(v_jit - v_ref) < 1e-3, f"V mismatch: {v_jit} vs {v_ref}"
        # V must be invariant under all 8 board symmetries
        array = np.array(fe.board_to_array(board))
        for rotations in range(4):
            for flip in (False, True):
                variant = np.rot90(array, rotations)
                if flip:
                    variant = np.fliplr(variant)
                v_sym = ntuple.evaluate(
                    U64(fe.board_from_array(variant)), weights, shifts, bases
                )
                assert abs(v_sym - v_jit) < 1e-3, "V not symmetry-invariant"
        moves = fe.afterstates(board)
        if not moves:
            board = fe.spawn(fe.spawn(0, py_rng), py_rng)
            continue
        _, board, _ = py_rng.choice(moves)
        if fe.count_empty(board):
            board = fe.spawn(board, py_rng)
    print("PASS value evaluation (jit == reference, symmetry-invariant, "
          "2,000 boards x 8 variants)")


def test_alpha_zero_is_noop():
    rng = np.random.default_rng(SEED + 1)
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)
    weights = rng.standard_normal(n_weights).astype(np.float32)
    snapshot = weights.copy()
    ej.seed_rng(SEED + 1)
    for _ in range(20):
        ntuple.play_training_episode(weights, shifts, bases, 0.0)
    assert np.array_equal(weights, snapshot), "alpha=0 episode modified weights"
    print("PASS alpha=0 episodes leave weights untouched (20 games)")


def test_training_determinism():
    # same seed + same starting weights => bitwise-identical weights after
    # training; any hidden state surviving between episodes would break this
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)
    runs = []
    for _ in range(2):
        weights = ntuple.new_weights(n_weights)
        ej.seed_rng(SEED + 2)
        stats = [
            ntuple.play_training_episode(weights, shifts, bases, 0.1)
            for _ in range(200)
        ]
        runs.append((weights, stats))
    assert np.array_equal(runs[0][0], runs[1][0]), "weights not reproducible"
    assert runs[0][1] == runs[1][1], "episode stats not reproducible"
    print("PASS training determinism (200 games, bitwise-identical weights)")


def test_training_learns(n_train=5_000, n_eval=500):
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)

    # baseline: greedy on all-zero weights = greedy on immediate merge score
    zero_weights = ntuple.new_weights(n_weights)
    ej.seed_rng(SEED + 3)
    baseline = [
        ntuple.play_greedy_episode(zero_weights, shifts, bases)[0]
        for _ in range(n_eval)
    ]
    baseline_avg = sum(baseline) / n_eval

    weights = ntuple.new_weights(n_weights)
    ej.seed_rng(SEED + 4)
    start = time.perf_counter()
    decisions = 0
    scores = []
    for _ in range(n_train):
        score, moves, _ = ntuple.play_training_episode(weights, shifts, bases, 0.1)
        decisions += moves
        scores.append(score)
    elapsed = time.perf_counter() - start

    ej.seed_rng(SEED + 5)
    trained = [
        ntuple.play_greedy_episode(weights, shifts, bases)[0]
        for _ in range(n_eval)
    ]
    trained_avg = sum(trained) / n_eval

    assert trained_avg > baseline_avg * 1.25 and trained_avg > 1_500, (
        f"training did not learn: baseline {baseline_avg:.0f}, "
        f"trained {trained_avg:.0f}"
    )
    print(f"PASS training learns (zero-weight greedy avg {baseline_avg:,.0f} "
          f"-> trained avg {trained_avg:,.0f} after {n_train:,} games)")
    print(f"     training throughput incl. V: {decisions / elapsed:,.0f} "
          f"decisions/sec ({n_train / elapsed:,.0f} games/sec)")


def test_tc_learning(n_train=5_000, n_eval=500):
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)

    # alpha = 0 must leave everything untouched
    weights = ntuple.new_weights(n_weights)
    e_acc, a_acc = ntuple.new_tc_state(n_weights)
    ej.seed_rng(SEED + 10)
    for _ in range(10):
        ntuple.play_training_episode_tc(weights, e_acc, a_acc, shifts, bases, 0.0)
    assert not np.any(weights), "alpha=0 TC episode changed weights"

    # rates must stay inside [0, 1] and accumulators stay consistent
    weights = ntuple.new_weights(n_weights)
    e_acc, a_acc = ntuple.new_tc_state(n_weights)
    ej.seed_rng(SEED + 11)
    scores = []
    for _ in range(n_train):
        score, _, _ = ntuple.play_training_episode_tc(
            weights, e_acc, a_acc, shifts, bases, 1.0
        )
        scores.append(score)
    touched = a_acc > 0
    assert np.all(np.abs(e_acc[touched]) <= a_acc[touched] * 1.001), (
        "|net error| exceeded total error magnitude"
    )
    assert np.all(np.isfinite(weights)), "TC produced non-finite weights"

    ej.seed_rng(SEED + 12)
    tc_eval = [
        ntuple.play_greedy_episode(weights, shifts, bases)[0] for _ in range(n_eval)
    ]
    tc_avg = sum(tc_eval) / n_eval

    # same budget under plain fixed-rate TD, for comparison
    plain = ntuple.new_weights(n_weights)
    ej.seed_rng(SEED + 11)
    for _ in range(n_train):
        ntuple.play_training_episode(plain, shifts, bases, 0.1)
    ej.seed_rng(SEED + 12)
    plain_eval = [
        ntuple.play_greedy_episode(plain, shifts, bases)[0] for _ in range(n_eval)
    ]
    plain_avg = sum(plain_eval) / n_eval

    assert tc_avg > 1_500, f"TC failed to learn: {tc_avg:.0f}"
    print(f"PASS temporal coherence ({n_train:,} games: TC avg {tc_avg:,.0f} "
          f"vs fixed-rate avg {plain_avg:,.0f})")


def test_staged_single_stage_equivalence(n_games=150):
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)
    no_thresholds = np.empty(0, dtype=np.int64)

    w_staged = ntuple.new_weights(n_weights)[None, :].copy()
    e_staged = np.zeros_like(w_staged)
    a_staged = np.zeros_like(w_staged)
    ej.seed_rng(SEED + 20)
    staged_stats = [
        ntuple.play_training_episode_staged(
            w_staged, e_staged, a_staged, no_thresholds, shifts, bases, 1.0
        )
        for _ in range(n_games)
    ]

    w_plain = ntuple.new_weights(n_weights)
    e_plain, a_plain = ntuple.new_tc_state(n_weights)
    ej.seed_rng(SEED + 20)
    plain_stats = [
        ntuple.play_training_episode_tc(
            w_plain, e_plain, a_plain, shifts, bases, 1.0
        )
        for _ in range(n_games)
    ]

    assert staged_stats == plain_stats, "episode stats diverged"
    assert np.array_equal(w_staged[0], w_plain), "weights diverged"
    assert np.array_equal(e_staged[0], e_plain)
    assert np.array_equal(a_staged[0], a_plain)
    print(f"PASS staged(no thresholds) == single-stage TC bitwise "
          f"({n_games} games)")


def test_staged_isolation_and_learning(n_games=3_000):
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)

    # an unreachable threshold must leave the upper stage untouched
    thresholds = np.array([15], dtype=np.int64)
    weights = np.zeros((2, n_weights), dtype=np.float32)
    e_acc = np.zeros_like(weights)
    a_acc = np.zeros_like(weights)
    ej.seed_rng(SEED + 21)
    for _ in range(200):
        ntuple.play_training_episode_staged(
            weights, e_acc, a_acc, thresholds, shifts, bases, 1.0
        )
    assert np.any(weights[0]), "stage 0 did not learn"
    assert not np.any(weights[1]), "unreachable stage was written to"
    assert not np.any(a_acc[1])

    # a reachable threshold must route late-game learning to the upper stage
    thresholds = np.array([10], dtype=np.int64)  # 1024
    weights = np.zeros((2, n_weights), dtype=np.float32)
    e_acc = np.zeros_like(weights)
    a_acc = np.zeros_like(weights)
    ej.seed_rng(SEED + 22)
    reached = 0
    for _ in range(n_games):
        _, _, max_exp = ntuple.play_training_episode_staged(
            weights, e_acc, a_acc, thresholds, shifts, bases, 1.0
        )
        if max_exp >= 10:
            reached += 1
    assert reached > 0, "no games reached the threshold; test is vacuous"
    assert np.any(weights[1]), "reachable stage never learned"
    print(f"PASS stage isolation + routing during training "
          f"({reached:,}/{n_games:,} games crossed the boundary)")


def test_staged_inference_routing():
    import os

    from models.NTupleAgentInference import NTupleAgentInference

    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)
    trained, shapes, games = ntuple.load_checkpoint(
        os.path.join("models", "ntuple_files", "ntuple_4cell_2shapes_final.npz")
    )
    zeros = ntuple.new_weights(n_weights)
    staged = np.stack([trained, zeros])
    path = os.path.join("models", "ntuple_files", "_test_staged.npz")
    ntuple.save_checkpoint(path, staged, shapes, games, 1.0, [],
                           stage_thresholds=(11,))
    agent = NTupleAgentInference(model_file_path=path)

    import random

    rng = random.Random(SEED + 23)
    low = np.zeros((4, 4), dtype=int)
    low[0] = [2, 4, 8, 16]
    high = low.copy()
    high[3][3] = 2048
    board_low = np.uint64(fe.board_from_array(low))
    board_high = np.uint64(fe.board_from_array(high))
    assert agent.get_action(obs_raw=low) == int(
        ntuple.choose_action(board_low, trained, shifts, bases)
    ), "below-threshold board did not use stage 0"
    assert agent.get_action(obs_raw=high) == int(
        ntuple.choose_action(board_high, zeros, shifts, bases)
    ), "above-threshold board did not use stage 1"

    # identical tables in every stage must reproduce the single-stage agent
    same = np.stack([trained, trained])
    ntuple.save_checkpoint(path, same, shapes, games, 1.0, [],
                           stage_thresholds=(11,))
    staged_agent = NTupleAgentInference(model_file_path=path, search_depth=2)
    single_agent = NTupleAgentInference(
        model_file_path=os.path.join(
            "models", "ntuple_files", "ntuple_4cell_2shapes_final.npz"
        ),
        search_depth=2,
    )
    board = fe.spawn(fe.spawn(0, rng), rng)
    for _ in range(200):
        array = np.array(fe.board_to_array(board))
        assert staged_agent.get_action(obs_raw=array) == single_agent.get_action(
            obs_raw=array
        )
        moves = fe.afterstates(board)
        if not moves:
            board = fe.spawn(fe.spawn(0, rng), rng)
            continue
        _, board, _ = rng.choice(moves)
        if fe.count_empty(board):
            board = fe.spawn(board, rng)
    os.remove(path)
    print("PASS staged inference routing (stage selection + search parity "
          "with single-stage agent)")


def test_checkpoint_roundtrip(tmp_name="models/ntuple_files/_test_ckpt.npz"):
    import os

    os.makedirs("models/ntuple_files", exist_ok=True)
    shifts, bases, n_weights = ntuple.build_placements(ntuple.SHAPES_4CELL)
    rng = np.random.default_rng(SEED + 6)
    weights = rng.standard_normal(n_weights).astype(np.float32)
    ntuple.save_checkpoint(
        tmp_name, weights, ntuple.SHAPES_4CELL, 123, 0.1, [10, 20]
    )
    loaded, shapes, games = ntuple.load_checkpoint(tmp_name)
    assert np.array_equal(loaded, weights)
    assert shapes == ntuple.SHAPES_4CELL and games == 123
    os.remove(tmp_name)
    print("PASS checkpoint save/load roundtrip")


def main():
    start = time.perf_counter()
    test_placement_expansion()
    test_6cell_shapes()
    test_choose_action()
    test_inference_agent()
    test_value_evaluation()
    test_alpha_zero_is_noop()
    test_training_determinism()
    test_training_learns()
    test_tc_learning()
    test_staged_single_stage_equivalence()
    test_staged_isolation_and_learning()
    test_staged_inference_routing()
    test_checkpoint_roundtrip()
    elapsed = time.perf_counter() - start
    print(f"\nALL TESTS PASSED in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

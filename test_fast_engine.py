"""Parity tests for fast_engine.py against envs/SGAI_2048.py.

Runnable directly (no pytest needed):  python test_fast_engine.py

Checks that slide results, merge scores, legality, spawns, and game-over
detection all match the environment exactly.
"""

import random
import time

import numpy as np

import fast_engine as fe
from envs.SGAI_2048 import SGAI_2048

SEED = 20260725

# Engine action ints are defined to match Act2048; map them to env methods
_ENV_SLIDE = {
    fe.UP: "_slide_up",
    fe.DOWN: "_slide_down",
    fe.LEFT: "_slide_left",
    fe.RIGHT: "_slide_right",
}

_PROFILES = ("sparse_low", "dense_mid", "high_tiles", "cascade_heavy", "anything")


def _random_board(rng, profile):
    """Random 4x4 array of tile VALUES in one of several stress profiles."""
    board = np.zeros((4, 4), dtype=int)
    if profile == "sparse_low":
        n_tiles, exponents = rng.randint(2, 8), range(1, 4)
    elif profile == "dense_mid":
        n_tiles, exponents = rng.randint(10, 16), range(1, 7)
    elif profile == "high_tiles":
        n_tiles, exponents = rng.randint(8, 16), range(8, 15)
    elif profile == "cascade_heavy":
        # long runs of equal small tiles maximize avalanche chains
        n_tiles, exponents = rng.randint(10, 16), (1, 1, 1, 2)
    elif profile == "anything":
        n_tiles, exponents = rng.randint(0, 16), range(1, 15)
    cells = rng.sample(range(16), n_tiles)
    for cell in cells:
        board[cell // 4][cell % 4] = 1 << rng.choice(exponents)
    return board


def _env_slide(env, array, action):
    """Run the environment's raw slide on a given board, no spawn, no reward."""
    env.board = array.copy()
    env.step_merge_score = 0
    getattr(env, _ENV_SLIDE[action])()
    return env.board, env.step_merge_score


def test_hand_traced_avalanche_cases():
    """Cascade cases traced by hand from the environment's loop semantics."""
    cases = [
        ([2, 2, 4, 0], fe.LEFT, [8, 0, 0, 0], 12),
        ([2, 2, 2, 2], fe.LEFT, [8, 0, 0, 0], 16),
        ([0, 4, 2, 2], fe.RIGHT, [0, 0, 0, 8], 12),
        ([2, 2, 2, 2], fe.RIGHT, [0, 0, 0, 8], 16),
    ]
    for row_values, action, expected_row, expected_score in cases:
        array = np.zeros((4, 4), dtype=int)
        array[0] = row_values
        board = fe.board_from_array(array)
        after, score, changed = fe.slide(board, action)
        result_row = fe.board_to_array(after)[0]
        assert result_row == expected_row, (
            f"{row_values} {action}: got {result_row}, want {expected_row}"
        )
        assert score == expected_score, (
            f"{row_values} {action}: score {score}, want {expected_score}"
        )
        assert changed
    # the same rows must cascade identically along columns (transpose path)
    for row_values, action, expected_row, expected_score in cases[:2]:
        array = np.zeros((4, 4), dtype=int)
        array[:, 0] = row_values
        board = fe.board_from_array(array)
        after, score, _ = fe.slide(board, fe.UP)
        result_col = [r[0] for r in fe.board_to_array(after)]
        assert result_col == expected_row and score == expected_score
    print("PASS hand-traced avalanche cases")


def test_transpose():
    rng = random.Random(SEED)
    for _ in range(10_000):
        board = rng.getrandbits(64)
        fast = fe.transpose(board)
        assert fast == fe._transpose_reference(board)
        assert fe.transpose(fast) == board  # transposing twice restores
    print("PASS transpose (10,000 random bit patterns)")


def test_slide_parity(boards_per_profile=2_400):
    """The non-negotiable test: engine == env on boards x all 4 actions."""
    rng = random.Random(SEED + 1)
    env = SGAI_2048()
    checked = 0
    overflow_checked = 0
    for profile in _PROFILES:
        for _ in range(boards_per_profile):
            array = _random_board(rng, profile)
            board = fe.board_from_array(array)
            for action in fe.ACTIONS:
                env_board, env_score = _env_slide(env, array, action)
                if env_board.max() > 32768:
                    # cascades can chain past 2**15 (e.g. four 16384s in a
                    # row); the engine must refuse these loudly
                    try:
                        fe.slide(board, action)
                        raise SystemExit(
                            f"FAIL: overflow slide did not raise:\n{array}"
                        )
                    except AssertionError:
                        overflow_checked += 1
                        continue
                after, score, changed = fe.slide(board, action)
                assert fe.board_to_array(after) == env_board.tolist(), (
                    f"board mismatch: profile={profile} action={action}\n"
                    f"input:\n{array}\nenv:\n{env_board}\n"
                    f"engine:\n{np.array(fe.board_to_array(after))}"
                )
                assert score == env_score, (
                    f"score mismatch: profile={profile} action={action} "
                    f"engine={score} env={env_score}\ninput:\n{array}"
                )
                assert changed == (not np.array_equal(array, env_board))
                checked += 1
            # game-over parity on the same board
            env.board = array.copy()
            env._is_game_over()
            assert fe.is_terminal(board) == env.game_over, (
                f"terminal mismatch on:\n{array}"
            )
    total_boards = boards_per_profile * len(_PROFILES)
    print(f"PASS slide parity ({total_boards:,} boards x 4 actions: "
          f"{checked:,} slides matched, {overflow_checked:,} overflow slides "
          f"correctly refused, plus game-over parity)")


def test_overflow_assertions():
    """Two 32768 tiles merging must fail loudly, never wrap silently."""
    array = np.zeros((4, 4), dtype=int)
    array[0][0] = array[0][1] = 32768
    board = fe.board_from_array(array)
    for action in (fe.LEFT, fe.RIGHT):
        try:
            fe.slide(board, action)
            raise SystemExit("FAIL: horizontal 32768 merge did not raise")
        except AssertionError:
            pass
    array = np.zeros((4, 4), dtype=int)
    array[0][0] = array[1][0] = 32768
    board = fe.board_from_array(array)
    for action in (fe.UP, fe.DOWN):
        try:
            fe.slide(board, action)
            raise SystemExit("FAIL: vertical 32768 merge did not raise")
        except AssertionError:
            pass
    # unrepresentable input values must also fail loudly
    for bad_value in (65536, 3):
        array = np.zeros((4, 4), dtype=int)
        array[2][2] = bad_value
        try:
            fe.board_from_array(array)
            raise SystemExit(f"FAIL: packing {bad_value} did not raise")
        except AssertionError:
            pass
    print("PASS overflow / bad-input assertions")


def test_engine_spawn_distribution(n=20_000):
    """Engine spawn: only empty cells, 2 vs 4 close to 50/50, cells uniform."""
    rng = random.Random(SEED + 2)
    array = np.zeros((4, 4), dtype=int)
    array[0] = [2, 4, 2, 4]  # top row occupied -> 12 empty cells
    board = fe.board_from_array(array)
    empty_cells = [i for i in range(16) if (board >> (4 * i)) & 0xF == 0]
    value_counts = {2: 0, 4: 0}
    cell_counts = dict.fromkeys(empty_cells, 0)
    for _ in range(n):
        spawned = fe.spawn(board, rng)
        diff = spawned ^ board
        assert diff != 0 and spawned & board == board  # one cell added, none changed
        cell = (diff.bit_length() - 1) // 4
        exponent = (spawned >> (4 * cell)) & 0xF
        assert cell in cell_counts, "spawned into an occupied cell"
        assert exponent in (1, 2), "spawned a value other than 2 or 4"
        value_counts[1 << exponent] += 1
        cell_counts[cell] += 1
    two_rate = value_counts[2] / n
    assert 0.46 < two_rate < 0.54, f"2-vs-4 rate off: {two_rate:.3f}"
    for cell, count in cell_counts.items():
        rate = count / n
        assert 0.065 < rate < 0.10, f"cell {cell} spawn rate off: {rate:.3f}"
    print(f"PASS engine spawn distribution ({n:,} spawns, "
          f"2-rate {two_rate:.3f})")


def test_env_step_spawn_behavior(n=5_000):
    """Env step(): a no-op move costs -10 AND still spawns exactly one tile."""
    env = SGAI_2048()
    noop_array = np.zeros((4, 4), dtype=int)
    noop_array[0] = [2, 4, 2, 4]  # nothing can move or merge upward
    value_counts = {2: 0, 4: 0}
    cell_counts = {}
    for _ in range(n):
        env.board = noop_array.copy()
        env.game_over = False
        obs, reward, terminated, truncated, _ = env.step(fe.UP)
        assert reward == -10, f"no-op reward was {reward}, want -10"
        assert not terminated
        diff = obs - noop_array
        changed_cells = np.argwhere(diff != 0)
        assert len(changed_cells) == 1, "no-op step must spawn exactly one tile"
        r, c = changed_cells[0]
        assert noop_array[r][c] == 0, "spawn landed on an occupied cell"
        assert obs[r][c] in (2, 4)
        value_counts[obs[r][c]] += 1
        cell_counts[(r, c)] = cell_counts.get((r, c), 0) + 1
    two_rate = value_counts[2] / n
    assert 0.45 < two_rate < 0.55, f"env 2-vs-4 rate off: {two_rate:.3f}"
    assert len(cell_counts) == 12, "spawns did not reach every empty cell"
    for cell, count in cell_counts.items():
        rate = count / n
        assert 0.055 < rate < 0.11, f"env cell {cell} spawn rate off: {rate:.3f}"
    print(f"PASS env one-spawn-per-step + no-op penalty ({n:,} steps, "
          f"2-rate {two_rate:.3f})")


def test_full_game_parity(n_games=150):
    """Drive real env games; the engine must track every step exactly."""
    rng = random.Random(SEED + 3)
    random.seed(SEED + 4)  # env spawns use the global random module
    env = SGAI_2048()
    total_steps = 0
    for _ in range(n_games):
        obs, _ = env.reset()
        board = fe.board_from_array(obs)
        reward_sum = 0
        highest = fe.max_tile(board)
        for _ in range(10_000):
            moves = fe.afterstates(board)
            legal = [m[0] for m in moves]
            illegal = [a for a in fe.ACTIONS if a not in legal]
            # mostly play legal moves; sometimes probe an illegal one on purpose
            if illegal and (not legal or rng.random() < 0.05):
                action = rng.choice(illegal)
                expected_after, expected_reward = board, -10
            else:
                action, expected_after, expected_reward = rng.choice(moves)
            obs, reward, terminated, truncated, _ = env.step(action)
            assert reward == expected_reward, (
                f"reward mismatch: env {reward}, engine {expected_reward}"
            )
            env_board = fe.board_from_array(obs)
            if fe.count_empty(expected_after) > 0:
                diff = env_board ^ expected_after
                assert diff != 0 and env_board & expected_after == expected_after, (
                    "env board is not our afterstate plus one spawned tile"
                )
                cell = (diff.bit_length() - 1) // 4
                spawned_exp = (env_board >> (4 * cell)) & 0xF
                assert (expected_after >> (4 * cell)) & 0xF == 0
                assert spawned_exp in (1, 2)
                assert diff == spawned_exp << (4 * cell), "more than one cell changed"
            else:
                assert env_board == expected_after, "spawn on a full board"
            board = env_board
            reward_sum += reward
            highest = max(highest, fe.max_tile(board))
            assert fe.is_terminal(board) == terminated
            total_steps += 1
            if terminated:
                break
        assert reward_sum == env.total_score, (
            f"score mismatch: engine {reward_sum}, env {env.total_score}"
        )
        assert highest == env.highest_tile
    print(f"PASS full-game parity ({n_games} games, {total_steps:,} steps: "
          f"boards, rewards, spawns, scores, terminals all match)")


def main():
    start = time.perf_counter()
    test_hand_traced_avalanche_cases()
    test_transpose()
    test_slide_parity()
    test_overflow_assertions()
    test_engine_spawn_distribution()
    test_env_step_spawn_behavior()
    test_full_game_parity()
    elapsed = time.perf_counter() - start
    print(f"\nALL TESTS PASSED in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

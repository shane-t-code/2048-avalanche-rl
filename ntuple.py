"""N-tuple value network with afterstate TD(0) learning.

V(board) = sum over base shapes, over all 8 symmetric placements of each
shape, of that shape's table entry indexed by the placement's cell exponents.
One weight table per base shape, concatenated into a single flat float32
array; placements carry their shape's base offset. Shapes must share one cell
count per network (4-bit index per cell -> table size 16**cells).

Training follows Szubert & Jaskowski-style afterstate TD(0), undiscounted,
pure greedy over legal moves (spawn randomness is the exploration).
"""

import numpy as np
from numba import njit

import engine_jit as ej

# base shapes as board cell indices (cell = 4*row + col)
SHAPES_4CELL = (
    (0, 1, 2, 3),  # top row line
    (0, 1, 4, 5),  # top-left 2x2 square
)

# 6-cell set: full-row-containing shapes, 2x3 rectangles, and edge/corner
# hooks. Shapes must come from distinct symmetry orbits — a shape that is a
# rotation/mirror of another would duplicate its table (test-enforced).
SHAPES_6CELL = (
    (0, 1, 2, 3, 4, 5),  # row 0 + (1,0) (1,1)
    (0, 1, 2, 3, 5, 6),  # row 0 + (1,1) (1,2)
    (0, 1, 2, 3, 4, 7),  # row 0 + (1,0) (1,3)
    (4, 5, 6, 7, 8, 9),  # row 1 + (2,0) (2,1)
    (4, 5, 6, 7, 9, 10),  # row 1 + (2,1) (2,2)
    (0, 1, 2, 4, 5, 6),  # 2x3 rectangle in the corner
    (4, 5, 6, 8, 9, 10),  # 2x3 rectangle on the edge, centered
    (0, 1, 2, 3, 7, 11),  # row 0 hooking down the right column
    (0, 1, 5, 6, 10, 11),  # staircase from the corner
    (0, 4, 5, 6, 7, 11),  # (0,0) + row 1 + (2,3): snake turn
)


def build_placements(shapes):
    """Expand base shapes into all 8 symmetric placements.

    Returns (shifts, bases, n_weights): shifts[p][j] is the bit offset of
    placement p's j-th cell, bases[p] is the flat-array offset of that
    placement's shape table.
    """
    cells_per_shape = len(shapes[0])
    for shape in shapes:
        if len(shape) != cells_per_shape:
            raise ValueError("all shapes in one network must share a cell count")
    grid = np.arange(16).reshape(4, 4)
    variants = []
    for rotations in range(4):
        g = np.rot90(grid, rotations)
        variants.append(g)
        variants.append(np.fliplr(g))
    table_size = 16**cells_per_shape
    n_placements = len(shapes) * len(variants)
    shifts = np.empty((n_placements, cells_per_shape), dtype=np.uint64)
    bases = np.empty(n_placements, dtype=np.int64)
    p = 0
    for shape_index, shape in enumerate(shapes):
        for g in variants:
            flat = g.reshape(16)
            for j, cell in enumerate(shape):
                shifts[p][j] = 4 * int(flat[cell])
            bases[p] = shape_index * table_size
            p += 1
    return shifts, bases, len(shapes) * table_size


def new_weights(n_weights):
    return np.zeros(n_weights, dtype=np.float32)


def new_tc_state(n_weights):
    """Per-weight temporal-coherence accumulators (net error, absolute error)."""
    return (
        np.zeros(n_weights, dtype=np.float32),
        np.zeros(n_weights, dtype=np.float32),
    )


@njit(cache=True)
def index_features(board, shifts, bases, out):
    """Write each placement's flat weight index for this board into out."""
    n_placements, cells = shifts.shape
    for p in range(n_placements):
        idx = np.int64(0)
        for j in range(cells):
            idx |= np.int64((board >> shifts[p][j]) & np.uint64(0xF)) << (4 * j)
        out[p] = bases[p] + idx
    return out


@njit(cache=True)
def evaluate(board, weights, shifts, bases):
    """V(board): sum of every placement's table entry."""
    n_placements, cells = shifts.shape
    total = 0.0
    for p in range(n_placements):
        idx = np.int64(0)
        for j in range(cells):
            idx |= np.int64((board >> shifts[p][j]) & np.uint64(0xF)) << (4 * j)
        total += weights[bases[p] + idx]
    return total


@njit(cache=True)
def play_training_episode(weights, shifts, bases, alpha):
    """One greedy TD(0) training game. Returns (score, moves, max_exponent).

    Per turn: compute all legal afterstates once, evaluate V on each, pick
    argmax(merge_reward + V), then update the previous afterstate toward
    r + V(chosen). Terminal target is 0. All per-episode learning state is
    local to this function, so nothing can leak across episodes.
    """
    n_placements = shifts.shape[0]
    alpha_eff = alpha / n_placements
    boards4 = np.empty(4, dtype=np.uint64)
    scores4 = np.empty(4, dtype=np.int64)
    actions4 = np.empty(4, dtype=np.int64)
    idx_buf = np.empty((4, n_placements), dtype=np.int64)
    prev_idx = np.empty(n_placements, dtype=np.int64)
    has_prev = False
    board = ej.spawn(ej.spawn(ej.ZERO))
    total_score = 0
    moves = 0
    while True:
        n = ej.afterstates_into(board, boards4, scores4, actions4)
        if n == 0:
            if has_prev:
                v_prev = 0.0
                for p in range(n_placements):
                    v_prev += weights[prev_idx[p]]
                delta = alpha_eff * (0.0 - v_prev)
                for p in range(n_placements):
                    weights[prev_idx[p]] += delta
            break
        best_k = 0
        best_v = 0.0
        best_val = -1e300
        for k in range(n):
            index_features(boards4[k], shifts, bases, idx_buf[k])
            v = 0.0
            for p in range(n_placements):
                v += weights[idx_buf[k][p]]
            val = scores4[k] + v
            if val > best_val:
                best_val = val
                best_v = v
                best_k = k
        if has_prev:
            v_prev = 0.0
            for p in range(n_placements):
                v_prev += weights[prev_idx[p]]
            delta = alpha_eff * (scores4[best_k] + best_v - v_prev)
            for p in range(n_placements):
                weights[prev_idx[p]] += delta
        for p in range(n_placements):
            prev_idx[p] = idx_buf[best_k][p]
        has_prev = True
        total_score += scores4[best_k]
        moves += 1
        board = ej.spawn(boards4[best_k])
    return total_score, moves, int(ej.max_exponent(board))


@njit(cache=True)
def play_training_episode_tc(weights, e_acc, a_acc, shifts, bases, alpha):
    """Training episode with temporal-coherence learning rates.

    Each weight's effective rate is |sum of its corrections| / |sum of their
    magnitudes|: weights whose corrections keep pulling the same direction
    keep learning fast, weights whose corrections cancel out settle down.
    Rates start at 1.0 for untouched weights.
    """
    n_placements = shifts.shape[0]
    share = 1.0 / n_placements
    boards4 = np.empty(4, dtype=np.uint64)
    scores4 = np.empty(4, dtype=np.int64)
    actions4 = np.empty(4, dtype=np.int64)
    idx_buf = np.empty((4, n_placements), dtype=np.int64)
    prev_idx = np.empty(n_placements, dtype=np.int64)
    has_prev = False
    board = ej.spawn(ej.spawn(ej.ZERO))
    total_score = 0
    moves = 0
    while True:
        n = ej.afterstates_into(board, boards4, scores4, actions4)
        if n == 0:
            if has_prev:
                v_prev = 0.0
                for p in range(n_placements):
                    v_prev += weights[prev_idx[p]]
                correction = (0.0 - v_prev) * share
                magnitude = abs(correction)
                for p in range(n_placements):
                    i = prev_idx[p]
                    denom = a_acc[i]
                    rate = 1.0 if denom == 0.0 else abs(e_acc[i]) / denom
                    weights[i] += alpha * rate * correction
                    e_acc[i] += correction
                    a_acc[i] += magnitude
            break
        best_k = 0
        best_v = 0.0
        best_val = -1e300
        for k in range(n):
            index_features(boards4[k], shifts, bases, idx_buf[k])
            v = 0.0
            for p in range(n_placements):
                v += weights[idx_buf[k][p]]
            val = scores4[k] + v
            if val > best_val:
                best_val = val
                best_v = v
                best_k = k
        if has_prev:
            v_prev = 0.0
            for p in range(n_placements):
                v_prev += weights[prev_idx[p]]
            correction = (scores4[best_k] + best_v - v_prev) * share
            magnitude = abs(correction)
            for p in range(n_placements):
                i = prev_idx[p]
                denom = a_acc[i]
                rate = 1.0 if denom == 0.0 else abs(e_acc[i]) / denom
                weights[i] += alpha * rate * correction
                e_acc[i] += correction
                a_acc[i] += magnitude
        for p in range(n_placements):
            prev_idx[p] = idx_buf[best_k][p]
        has_prev = True
        total_score += scores4[best_k]
        moves += 1
        board = ej.spawn(boards4[best_k])
    return total_score, moves, int(ej.max_exponent(board))


@njit(cache=True)
def play_training_episode_staged(weights2, e2, a2, thresholds, shifts, bases,
                                 alpha):
    """Multi-stage training episode with temporal-coherence rates.

    weights2/e2/a2 are (n_stages, n_weights); thresholds is an ascending
    int64 array of stage-entry max exponents (stage s applies once the
    board's max exponent reaches thresholds[s-1]). Each move evaluates and
    learns in the stage of the current board; the previous afterstate is
    updated in its own stage's tables toward the current stage's value, so
    values hand off cleanly at stage boundaries. An empty thresholds array
    reproduces single-stage TC training exactly.
    """
    n_placements = shifts.shape[0]
    n_thresholds = thresholds.shape[0]
    share = 1.0 / n_placements
    boards4 = np.empty(4, dtype=np.uint64)
    scores4 = np.empty(4, dtype=np.int64)
    actions4 = np.empty(4, dtype=np.int64)
    idx_buf = np.empty((4, n_placements), dtype=np.int64)
    prev_idx = np.empty(n_placements, dtype=np.int64)
    has_prev = False
    prev_stage = 0
    stage = 0
    board = ej.spawn(ej.spawn(ej.ZERO))
    total_score = 0
    moves = 0
    while True:
        n = ej.afterstates_into(board, boards4, scores4, actions4)
        if n == 0:
            if has_prev:
                weights = weights2[prev_stage]
                e_acc = e2[prev_stage]
                a_acc = a2[prev_stage]
                v_prev = 0.0
                for p in range(n_placements):
                    v_prev += weights[prev_idx[p]]
                correction = (0.0 - v_prev) * share
                magnitude = abs(correction)
                for p in range(n_placements):
                    i = prev_idx[p]
                    denom = a_acc[i]
                    rate = 1.0 if denom == 0.0 else abs(e_acc[i]) / denom
                    weights[i] += alpha * rate * correction
                    e_acc[i] += correction
                    a_acc[i] += magnitude
            break
        max_exp = np.int64(ej.max_exponent(board))
        while stage < n_thresholds and max_exp >= thresholds[stage]:
            stage += 1
        weights = weights2[stage]
        best_k = 0
        best_v = 0.0
        best_val = -1e300
        for k in range(n):
            index_features(boards4[k], shifts, bases, idx_buf[k])
            v = 0.0
            for p in range(n_placements):
                v += weights[idx_buf[k][p]]
            val = scores4[k] + v
            if val > best_val:
                best_val = val
                best_v = v
                best_k = k
        if has_prev:
            weights_prev = weights2[prev_stage]
            e_acc = e2[prev_stage]
            a_acc = a2[prev_stage]
            v_prev = 0.0
            for p in range(n_placements):
                v_prev += weights_prev[prev_idx[p]]
            correction = (scores4[best_k] + best_v - v_prev) * share
            magnitude = abs(correction)
            for p in range(n_placements):
                i = prev_idx[p]
                denom = a_acc[i]
                rate = 1.0 if denom == 0.0 else abs(e_acc[i]) / denom
                weights_prev[i] += alpha * rate * correction
                e_acc[i] += correction
                a_acc[i] += magnitude
        for p in range(n_placements):
            prev_idx[p] = idx_buf[best_k][p]
        prev_stage = stage
        has_prev = True
        total_score += scores4[best_k]
        moves += 1
        board = ej.spawn(boards4[best_k])
    return total_score, moves, int(ej.max_exponent(board))


def stage_for_exponent(max_exp, thresholds):
    """Stage index for a board whose largest tile has this exponent."""
    stage = 0
    for threshold in thresholds:
        if max_exp >= threshold:
            stage += 1
    return stage


@njit(cache=True)
def choose_action(board, weights, shifts, bases):
    """Greedy legal action for a board: argmax merge_reward + V(afterstate).

    Returns the action int, or -1 when no move changes the board (terminal).
    Illegal actions can never be returned.
    """
    boards4 = np.empty(4, dtype=np.uint64)
    scores4 = np.empty(4, dtype=np.int64)
    actions4 = np.empty(4, dtype=np.int64)
    n = ej.afterstates_into(board, boards4, scores4, actions4)
    if n == 0:
        return np.int64(-1)
    best_k = 0
    best_val = -1e300
    for k in range(n):
        val = scores4[k] + evaluate(boards4[k], weights, shifts, bases)
        if val > best_val:
            best_val = val
            best_k = k
    return actions4[best_k]


@njit(cache=True)
def play_greedy_episode(weights, shifts, bases):
    """One greedy game with no learning. Returns (score, moves, max_exponent)."""
    n_placements = shifts.shape[0]
    boards4 = np.empty(4, dtype=np.uint64)
    scores4 = np.empty(4, dtype=np.int64)
    actions4 = np.empty(4, dtype=np.int64)
    board = ej.spawn(ej.spawn(ej.ZERO))
    total_score = 0
    moves = 0
    while True:
        n = ej.afterstates_into(board, boards4, scores4, actions4)
        if n == 0:
            break
        best_k = 0
        best_val = -1e300
        for k in range(n):
            val = scores4[k] + evaluate(boards4[k], weights, shifts, bases)
            if val > best_val:
                best_val = val
                best_k = k
        total_score += scores4[best_k]
        moves += 1
        board = ej.spawn(boards4[best_k])
    return total_score, moves, int(ej.max_exponent(board))


def save_checkpoint(path, weights, shapes, games_trained, alpha, recent_scores,
                    stage_thresholds=None):
    arrays = {
        "weights": weights,
        "shapes": np.array(shapes, dtype=np.int64),
        "games_trained": np.int64(games_trained),
        "alpha": np.float64(alpha),
        "recent_scores": np.array(recent_scores, dtype=np.int64),
    }
    if stage_thresholds is not None:
        arrays["stage_thresholds"] = np.array(stage_thresholds, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def load_checkpoint(path):
    data = np.load(path)
    shapes = tuple(tuple(int(c) for c in row) for row in data["shapes"])
    return data["weights"], shapes, int(data["games_trained"])


def load_checkpoint_full(path):
    """Like load_checkpoint but also returns stage thresholds (or None)."""
    data = np.load(path)
    shapes = tuple(tuple(int(c) for c in row) for row in data["shapes"])
    thresholds = None
    if "stage_thresholds" in data:
        thresholds = tuple(int(t) for t in data["stage_thresholds"])
    return data["weights"], shapes, int(data["games_trained"]), thresholds

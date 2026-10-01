"""Expectimax search over the n-tuple value function.

Max nodes pick a move (illegal moves are never candidates); each move yields
its afterstate plus merge reward; chance nodes expand the spawn — every empty
cell times tile values 2 and 4 at equal probability, matching this game's
spawn rule. Leaves score accumulated reward + V(afterstate). Depth counts max
levels: depth 1 is exactly the greedy policy (test-enforced).

The transposition table is open-addressed on the packed board, exact-match on
remaining depth, invalidated per search via a generation stamp; it caches
exact values, so results with the table on and off are identical
(test-enforced). Chance nodes with more empty cells than chance_cap expand a
deterministic evenly-strided subset with renormalized weights.

_state_value inlines its chance node because numba supports self-recursion
only; _chance_value is a root-level helper and must stay operation-identical
to the inlined block.
"""

import time

import numpy as np
from numba import njit

import engine_jit as ej
import ntuple

_HASH_MULT = np.uint64(0x9E3779B97F4A7C15)
_M_CELL = np.uint64(0xF)
_ZERO = np.uint64(0)
_PROBES = 4


@njit(cache=True)
def _tt_lookup(board, depth, tt_keys, tt_depths, tt_gens, tt_vals,
               tt_shift, tt_mask, gen):
    if tt_mask == 0:
        return False, 0.0
    slot = np.int64((board * _HASH_MULT) >> tt_shift)
    for probe in range(_PROBES):
        s = (slot + probe) & tt_mask
        if tt_gens[s] == gen and tt_keys[s] == board and tt_depths[s] == depth:
            return True, tt_vals[s]
    return False, 0.0


@njit(cache=True)
def _tt_store(board, depth, value, tt_keys, tt_depths, tt_gens, tt_vals,
              tt_shift, tt_mask, gen):
    if tt_mask == 0:
        return
    slot = np.int64((board * _HASH_MULT) >> tt_shift)
    target = slot & tt_mask
    for probe in range(_PROBES):
        s = (slot + probe) & tt_mask
        if tt_gens[s] != gen:
            target = s
            break
    tt_keys[target] = board
    tt_depths[target] = depth
    tt_gens[target] = gen
    tt_vals[target] = value


@njit(cache=True)
def _state_value(state, depth, weights, shifts, bases, chance_cap,
                 tt_keys, tt_depths, tt_gens, tt_vals, tt_shift, tt_mask, gen):
    """Best over legal moves of merge reward + future value; 0 at terminal."""
    boards4 = np.empty(4, dtype=np.uint64)
    scores4 = np.empty(4, dtype=np.int64)
    actions4 = np.empty(4, dtype=np.int64)
    n = ej.afterstates_into(state, boards4, scores4, actions4)
    if n == 0:
        return 0.0
    best = -1e300
    for k in range(n):
        if depth <= 1:
            val = scores4[k] + ntuple.evaluate(boards4[k], weights, shifts, bases)
        else:
            after = boards4[k]
            empty_shifts = np.empty(16, dtype=np.uint64)
            m = 0
            for i in range(16):
                shift = np.uint64(4 * i)
                if (after >> shift) & _M_CELL == _ZERO:
                    empty_shifts[m] = shift
                    m += 1
            n_expand = m if m <= chance_cap else chance_cap
            total = 0.0
            for j in range(n_expand):
                cell_shift = empty_shifts[(j * m) // n_expand]
                for exp in range(1, 3):
                    child = after | (np.uint64(exp) << cell_shift)
                    found, value = _tt_lookup(
                        child, depth - 1, tt_keys, tt_depths, tt_gens,
                        tt_vals, tt_shift, tt_mask, gen,
                    )
                    if not found:
                        value = _state_value(
                            child, depth - 1, weights, shifts, bases,
                            chance_cap, tt_keys, tt_depths, tt_gens, tt_vals,
                            tt_shift, tt_mask, gen,
                        )
                        _tt_store(
                            child, depth - 1, value, tt_keys, tt_depths,
                            tt_gens, tt_vals, tt_shift, tt_mask, gen,
                        )
                    total += value
            val = scores4[k] + total / (2.0 * n_expand)
        if val > best:
            best = val
    return best


@njit(cache=True)
def _chance_value(after, depth, weights, shifts, bases, chance_cap,
                  tt_keys, tt_depths, tt_gens, tt_vals, tt_shift, tt_mask,
                  gen):
    """Expected next-state value over the spawn; root-level helper."""
    empty_shifts = np.empty(16, dtype=np.uint64)
    m = 0
    for i in range(16):
        shift = np.uint64(4 * i)
        if (after >> shift) & _M_CELL == _ZERO:
            empty_shifts[m] = shift
            m += 1
    n_expand = m if m <= chance_cap else chance_cap
    total = 0.0
    for j in range(n_expand):
        cell_shift = empty_shifts[(j * m) // n_expand]
        for exp in range(1, 3):
            child = after | (np.uint64(exp) << cell_shift)
            found, value = _tt_lookup(
                child, depth - 1, tt_keys, tt_depths, tt_gens, tt_vals,
                tt_shift, tt_mask, gen,
            )
            if not found:
                value = _state_value(
                    child, depth - 1, weights, shifts, bases, chance_cap,
                    tt_keys, tt_depths, tt_gens, tt_vals, tt_shift, tt_mask,
                    gen,
                )
                _tt_store(
                    child, depth - 1, value, tt_keys, tt_depths, tt_gens,
                    tt_vals, tt_shift, tt_mask, gen,
                )
            total += value
    return total / (2.0 * n_expand)


@njit(cache=True)
def search_best_action(state, depth, weights, shifts, bases, chance_cap,
                       tt_keys, tt_depths, tt_gens, tt_vals, tt_shift,
                       tt_mask, gen):
    """Best action for a state, or -1 when no move changes the board."""
    boards4 = np.empty(4, dtype=np.uint64)
    scores4 = np.empty(4, dtype=np.int64)
    actions4 = np.empty(4, dtype=np.int64)
    n = ej.afterstates_into(state, boards4, scores4, actions4)
    if n == 0:
        return np.int64(-1)
    best_k = 0
    best_val = -1e300
    for k in range(n):
        if depth <= 1:
            val = scores4[k] + ntuple.evaluate(boards4[k], weights, shifts, bases)
        else:
            val = scores4[k] + _chance_value(
                boards4[k], depth, weights, shifts, bases, chance_cap,
                tt_keys, tt_depths, tt_gens, tt_vals, tt_shift, tt_mask, gen,
            )
        if val > best_val:
            best_val = val
            best_k = k
    return actions4[best_k]


class ExpectimaxSearch:
    """Configurable search: depth, chance-node cap, optional per-move time
    budget (iterative deepening, checked between depths), optional extra
    depth when empty cells run low, and a transposition table (tt_log2 = 0
    disables it)."""

    def __init__(self, weights, shifts, bases, depth=2, chance_cap=6,
                 time_budget_ms=None, deepen_empty_threshold=0,
                 deepen_extra=0, tt_log2=20):
        self.weights = weights
        self.shifts = shifts
        self.bases = bases
        self.depth = depth
        self.chance_cap = chance_cap
        self.time_budget_ms = time_budget_ms
        self.deepen_empty_threshold = deepen_empty_threshold
        self.deepen_extra = deepen_extra
        size = 1 << tt_log2 if tt_log2 > 0 else 1
        self.tt_keys = np.zeros(size, dtype=np.uint64)
        self.tt_depths = np.zeros(size, dtype=np.int8)
        self.tt_gens = np.zeros(size, dtype=np.uint32)
        self.tt_vals = np.zeros(size, dtype=np.float64)
        self.tt_shift = np.uint64(64 - tt_log2) if tt_log2 > 0 else np.uint64(63)
        self.tt_mask = np.int64(size - 1) if tt_log2 > 0 else np.int64(0)
        self.gen = 0

    def _run(self, board, depth):
        self.gen += 1
        return int(search_best_action(
            board, depth, self.weights, self.shifts, self.bases,
            self.chance_cap, self.tt_keys, self.tt_depths, self.tt_gens,
            self.tt_vals, self.tt_shift, self.tt_mask, np.uint32(self.gen),
        ))

    def best_action(self, board):
        board = np.uint64(board)
        depth = self.depth
        if self.deepen_extra and ej.count_empty(board) <= self.deepen_empty_threshold:
            depth += self.deepen_extra
        if self.time_budget_ms is None:
            return self._run(board, depth)
        deadline = time.perf_counter() + self.time_budget_ms / 1000.0
        action = self._run(board, 1)
        d = 2
        while d <= depth and time.perf_counter() < deadline:
            action = self._run(board, d)
            d += 1
        return action

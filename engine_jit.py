"""Numba-jitted port of fast_engine.

Boards are np.uint64 end-to-end. Signed shifts sign-extend once the top-corner
cell holds a large tile, so every value stays unsigned and constants are
np.uint64; do not mix plain int arithmetic into the board math. Callers
beware: these functions RETURN boards as plain Python ints — wrap with
np.uint64 before passing one back in, or boards using bit 63 (a tile of 256+
in the last cell) overflow the implicit int64 argument conversion. Parity
with fast_engine is enforced by test_engine_jit.py.
"""

import numpy as np
from numba import njit

import fast_engine as _fe

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

LEFT_TABLE = np.array(_fe._LEFT, dtype=np.uint64)
RIGHT_TABLE = np.array(_fe._RIGHT, dtype=np.uint64)
SCORE_LEFT = np.array(_fe._SCORE_LEFT, dtype=np.int64)
SCORE_RIGHT = np.array(_fe._SCORE_RIGHT, dtype=np.int64)

ZERO = np.uint64(0)
_M_ROW = np.uint64(0xFFFF)
_M_CELL = np.uint64(0xF)
_S16 = np.uint64(16)
_S32 = np.uint64(32)
_S48 = np.uint64(48)
_S12, _S24, _S36 = np.uint64(12), np.uint64(24), np.uint64(36)
_DIAG = np.uint64(0xF000_0F00_00F0_000F)
_UP1 = np.uint64(0x0000_F000_0F00_00F0)
_UP2 = np.uint64(0x0000_0000_F000_0F00)
_UP3 = np.uint64(0x0000_0000_0000_F000)
_DN1 = np.uint64(0x0F00_00F0_000F_0000)
_DN2 = np.uint64(0x00F0_000F_0000_0000)
_DN3 = np.uint64(0x000F_0000_0000_0000)


@njit(cache=True)
def seed_rng(seed):
    # numba maintains its own np.random state; this seeds the jitted side
    np.random.seed(seed)


@njit(cache=True)
def transpose(b):
    return (
        (b & _DIAG)
        | ((b & _UP1) << _S12)
        | ((b & _UP2) << _S24)
        | ((b & _UP3) << _S36)
        | ((b & _DN1) >> _S12)
        | ((b & _DN2) >> _S24)
        | ((b & _DN3) >> _S36)
    )


@njit(cache=True)
def _slide_rows_left(b):
    r0 = b & _M_ROW
    r1 = (b >> _S16) & _M_ROW
    r2 = (b >> _S32) & _M_ROW
    r3 = b >> _S48
    score = SCORE_LEFT[r0] + SCORE_LEFT[r1] + SCORE_LEFT[r2] + SCORE_LEFT[r3]
    if score < 0:
        raise AssertionError(
            "slide would create a tile above 32768 (2**15), which the 4-bit "
            "board packing cannot represent"
        )
    nb = (
        LEFT_TABLE[r0]
        | (LEFT_TABLE[r1] << _S16)
        | (LEFT_TABLE[r2] << _S32)
        | (LEFT_TABLE[r3] << _S48)
    )
    return nb, score


@njit(cache=True)
def _slide_rows_right(b):
    r0 = b & _M_ROW
    r1 = (b >> _S16) & _M_ROW
    r2 = (b >> _S32) & _M_ROW
    r3 = b >> _S48
    score = SCORE_RIGHT[r0] + SCORE_RIGHT[r1] + SCORE_RIGHT[r2] + SCORE_RIGHT[r3]
    if score < 0:
        raise AssertionError(
            "slide would create a tile above 32768 (2**15), which the 4-bit "
            "board packing cannot represent"
        )
    nb = (
        RIGHT_TABLE[r0]
        | (RIGHT_TABLE[r1] << _S16)
        | (RIGHT_TABLE[r2] << _S32)
        | (RIGHT_TABLE[r3] << _S48)
    )
    return nb, score


@njit(cache=True)
def slide(b, action):
    """Apply one move without spawning. Returns (afterstate, merge_score, changed)."""
    if action == 2:
        nb, score = _slide_rows_left(b)
    elif action == 3:
        nb, score = _slide_rows_right(b)
    elif action == 0:
        t = transpose(b)
        nt, score = _slide_rows_left(t)
        nb = transpose(nt)
    else:
        t = transpose(b)
        nt, score = _slide_rows_right(t)
        nb = transpose(nt)
    return nb, score, nb != b


@njit(cache=True)
def afterstates_into(board, boards4, scores4, actions4):
    """Fill the buffers with all legal moves in UP, DOWN, LEFT, RIGHT order.

    Returns the number of legal moves. Buffers must be length-4 arrays of
    dtype uint64 / int64 / int64.
    """
    n = 0
    t = transpose(board)
    nt, s = _slide_rows_left(t)
    if nt != t:
        boards4[n] = transpose(nt)
        scores4[n] = s
        actions4[n] = 0
        n += 1
    nt, s = _slide_rows_right(t)
    if nt != t:
        boards4[n] = transpose(nt)
        scores4[n] = s
        actions4[n] = 1
        n += 1
    nb, s = _slide_rows_left(board)
    if nb != board:
        boards4[n] = nb
        scores4[n] = s
        actions4[n] = 2
        n += 1
    nb, s = _slide_rows_right(board)
    if nb != board:
        boards4[n] = nb
        scores4[n] = s
        actions4[n] = 3
        n += 1
    return n


@njit(cache=True)
def spawn(board):
    """Add the per-turn random tile: 2 or 4 at 50/50, uniform over empty cells."""
    n = 0
    for i in range(16):
        if (board >> np.uint64(4 * i)) & _M_CELL == ZERO:
            n += 1
    k = np.random.randint(0, n)
    seen = 0
    for i in range(16):
        if (board >> np.uint64(4 * i)) & _M_CELL == ZERO:
            if seen == k:
                exp = np.uint64(1) if np.random.random() < 0.5 else np.uint64(2)
                return board | (exp << np.uint64(4 * i))
            seen += 1
    return board


@njit(cache=True)
def count_empty(board):
    n = 0
    for i in range(16):
        if (board >> np.uint64(4 * i)) & _M_CELL == ZERO:
            n += 1
    return n


@njit(cache=True)
def max_exponent(board):
    best = np.uint64(0)
    b = board
    for _ in range(16):
        e = b & _M_CELL
        if e > best:
            best = e
        b >>= np.uint64(4)
    return best

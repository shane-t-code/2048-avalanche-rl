"""Bitboard engine for 2048 Avalanche.

Pure functions, no pygame/gym dependencies. A board is one integer: 16 cells
x 4 bits, each holding a tile exponent (0 = empty, k = tile 2**k). Per-row
slide results and merge scores for all 65536 possible rows are precomputed at
import. Slide semantics must match envs/SGAI_2048.py exactly; parity is
enforced by test_fast_engine.py.

Cell (r, c) lives at bit offset 16*r + 4*c.
"""

import random

# must match Act2048 in envs/SGAI_2048.py
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)

ROW_MASK = 0xFFFF
CELL_MASK = 0xF
MAX_EXPONENT = 15  # 4-bit cells cap tiles at 2**15 = 32768

# planted in the score tables for rows whose slide would create a tile above
# 2**15; makes the summed score negative so slide() fails loudly instead of
# silently wrapping the exponent
_OVERFLOW_SENTINEL = -(1 << 40)


def _row_slide_left_values(values):
    # transcribed loop-for-loop from SGAI_2048._slide_left (single row) so the
    # tables inherit its exact cascading-merge semantics; do not "clean up"
    row = list(values)
    score = 0
    for main in range(1, 4):
        if row[main] == 0:
            continue
        for inner in range(main - 1, -1, -1):
            if row[inner] == 0:
                row[inner] = row[inner + 1]
                row[inner + 1] = 0
            elif row[inner] == row[inner + 1]:
                row[inner] = row[inner] * 2
                score += row[inner]
                row[inner + 1] = 0
            else:
                continue
    return row, score


def _row_slide_right_values(values):
    # transcribed from SGAI_2048._slide_right; same fidelity requirement
    row = list(values)
    score = 0
    for main in range(3, -1, -1):
        if row[main] == 0:
            continue
        for inner in range(main + 1, 4):
            if row[inner] == 0:
                row[inner] = row[inner - 1]
                row[inner - 1] = 0
            elif row[inner] == row[inner - 1]:
                row[inner] = row[inner] * 2
                score += row[inner]
                row[inner - 1] = 0
            else:
                continue
    return row, score


def _unpack_row(row):
    values = []
    for i in range(4):
        exp = (row >> (4 * i)) & CELL_MASK
        values.append(0 if exp == 0 else 1 << exp)
    return values


def _pack_row(values):
    row = 0
    overflow = False
    for i, v in enumerate(values):
        if v == 0:
            continue
        exp = v.bit_length() - 1
        if exp > MAX_EXPONENT:
            overflow = True
            exp &= CELL_MASK  # placeholder; slide() raises before this is used
        row |= exp << (4 * i)
    return row, overflow


def _build_tables():
    left = [0] * 65536
    right = [0] * 65536
    score_left = [0] * 65536
    score_right = [0] * 65536
    can_merge = bytearray(65536)
    for packed in range(65536):
        values = _unpack_row(packed)

        new_values, score = _row_slide_left_values(values)
        new_packed, overflow = _pack_row(new_values)
        left[packed] = new_packed
        score_left[packed] = _OVERFLOW_SENTINEL if overflow else score

        new_values, score = _row_slide_right_values(values)
        new_packed, overflow = _pack_row(new_values)
        right[packed] = new_packed
        score_right[packed] = _OVERFLOW_SENTINEL if overflow else score

        can_merge[packed] = int(
            any(values[i] != 0 and values[i] == values[i + 1] for i in range(3))
        )
    return left, right, score_left, score_right, can_merge


_LEFT, _RIGHT, _SCORE_LEFT, _SCORE_RIGHT, _CAN_MERGE = _build_tables()


# transpose swaps cell (r, c) with (c, r): bit offset 16r+4c -> 16c+4r, a
# shift of 12*(c-r); cells grouped by (c-r) shift together
_DIAG = 0xF000_0F00_00F0_000F
_UP1 = 0x0000_F000_0F00_00F0
_UP2 = 0x0000_0000_F000_0F00
_UP3 = 0x0000_0000_0000_F000
_DN1 = 0x0F00_00F0_000F_0000
_DN2 = 0x00F0_000F_0000_0000
_DN3 = 0x000F_0000_0000_0000


def transpose(board):
    """Swap rows and columns."""
    return (
        (board & _DIAG)
        | ((board & _UP1) << 12)
        | ((board & _UP2) << 24)
        | ((board & _UP3) << 36)
        | ((board & _DN1) >> 12)
        | ((board & _DN2) >> 24)
        | ((board & _DN3) >> 36)
    )


def _transpose_reference(board):
    # slow nibble-by-nibble version, used only by tests to verify transpose()
    result = 0
    for r in range(4):
        for c in range(4):
            nibble = (board >> (16 * r + 4 * c)) & CELL_MASK
            result |= nibble << (16 * c + 4 * r)
    return result


def _slide_rows(board, table, score_table):
    r0 = board & ROW_MASK
    r1 = (board >> 16) & ROW_MASK
    r2 = (board >> 32) & ROW_MASK
    r3 = board >> 48
    score = score_table[r0] + score_table[r1] + score_table[r2] + score_table[r3]
    if score < 0:
        raise AssertionError(
            "slide would create a tile above 32768 (2**15), which the 4-bit "
            "board packing cannot represent"
        )
    new_board = table[r0] | (table[r1] << 16) | (table[r2] << 32) | (table[r3] << 48)
    return new_board, score


def slide(board, action):
    """Apply one move without spawning. Returns (afterstate, merge_score, changed)."""
    if action == LEFT:
        new_board, score = _slide_rows(board, _LEFT, _SCORE_LEFT)
    elif action == RIGHT:
        new_board, score = _slide_rows(board, _RIGHT, _SCORE_RIGHT)
    elif action == UP:
        t = transpose(board)
        new_t, score = _slide_rows(t, _LEFT, _SCORE_LEFT)
        new_board = transpose(new_t)
    elif action == DOWN:
        t = transpose(board)
        new_t, score = _slide_rows(t, _RIGHT, _SCORE_RIGHT)
        new_board = transpose(new_t)
    else:
        raise ValueError(f"invalid action {action!r}")
    return new_board, score, new_board != board


def afterstates(board):
    """All legal moves as a list of (action, afterstate, merge_score).

    Computes each slide once so callers can reuse the results for action
    selection, learning, and legality checks. UP and DOWN share one forward
    transpose; the back-transpose runs only for moves that changed the board.
    """
    result = []

    new_board, score = _slide_rows(board, _LEFT, _SCORE_LEFT)
    if new_board != board:
        result.append((LEFT, new_board, score))
    new_board, score = _slide_rows(board, _RIGHT, _SCORE_RIGHT)
    if new_board != board:
        result.append((RIGHT, new_board, score))

    t = transpose(board)
    new_t, score = _slide_rows(t, _LEFT, _SCORE_LEFT)
    if new_t != t:
        result.append((UP, transpose(new_t), score))
    new_t, score = _slide_rows(t, _RIGHT, _SCORE_RIGHT)
    if new_t != t:
        result.append((DOWN, transpose(new_t), score))

    result.sort()  # deterministic UP, DOWN, LEFT, RIGHT order
    return result


def legal_actions(board):
    """Actions that would change the board."""
    return [action for action, _, _ in afterstates(board)]


def spawn(board, rng=random):
    """Add the per-turn random tile: 2 or 4 at 50/50, uniform over empty cells."""
    empties = [i for i in range(16) if (board >> (4 * i)) & CELL_MASK == 0]
    cell = rng.choice(empties)
    exponent = rng.choice((1, 2))
    return board | (exponent << (4 * cell))


def count_empty(board):
    """Number of empty cells."""
    count = 0
    for i in range(16):
        if (board >> (4 * i)) & CELL_MASK == 0:
            count += 1
    return count


def is_terminal(board):
    """True when no move can change the board; mirrors SGAI_2048._is_game_over."""
    b = board
    for _ in range(16):
        if b & CELL_MASK == 0:
            return False
        b >>= 4
    t = transpose(board)
    for shift in (0, 16, 32, 48):
        if _CAN_MERGE[(board >> shift) & ROW_MASK]:
            return False
        if _CAN_MERGE[(t >> shift) & ROW_MASK]:
            return False
    return True


def max_tile(board):
    """Largest tile value on the board (0 if empty)."""
    best = 0
    b = board
    for _ in range(16):
        exp = b & CELL_MASK
        if exp > best:
            best = exp
        b >>= 4
    return 0 if best == 0 else 1 << best


def board_from_array(array):
    """Pack a 4x4 array of tile values (env observation format) into a board."""
    board = 0
    for r in range(4):
        for c in range(4):
            value = int(array[r][c])
            if value == 0:
                continue
            exponent = value.bit_length() - 1
            if (1 << exponent) != value or not 1 <= exponent <= MAX_EXPONENT:
                raise AssertionError(
                    f"cell ({r},{c}) holds {value}, not a power of two in "
                    f"[2, 32768]"
                )
            board |= exponent << (4 * (4 * r + c))
    return board


def board_to_array(board):
    """Unpack a board into a 4x4 list-of-lists of tile values."""
    array = []
    for r in range(4):
        row = []
        for c in range(4):
            exp = (board >> (4 * (4 * r + c))) & CELL_MASK
            row.append(0 if exp == 0 else 1 << exp)
        array.append(row)
    return array

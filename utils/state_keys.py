import numpy as np

# Act2048 action indices (UP=0, DOWN=1, LEFT=2, RIGHT=3), duplicated here as plain
# ints to avoid a circular import with envs.SGAI_2048
_UP, _DOWN, _LEFT, _RIGHT = 0, 1, 2, 3

# One counterclockwise np.rot90 moves the top edge of the board to the left edge,
# so a tile sliding UP on the original board slides LEFT on the rotated board,
# and so on around the compass
_ROTATE_ACTION = {_UP: _LEFT, _LEFT: _DOWN, _DOWN: _RIGHT, _RIGHT: _UP}
# np.fliplr mirrors the columns: LEFT and RIGHT swap, UP and DOWN are unchanged
_FLIP_ACTION = {_UP: _UP, _DOWN: _DOWN, _LEFT: _RIGHT, _RIGHT: _LEFT}


def _action_map(num_rotations: int, flipped: bool) -> dict:
    """Map each original action index to its equivalent on the transformed board."""
    mapping = {}
    for action in (_UP, _DOWN, _LEFT, _RIGHT):
        mapped = action
        for _ in range(num_rotations):
            mapped = _ROTATE_ACTION[mapped]
        if flipped:
            mapped = _FLIP_ACTION[mapped]
        mapping[action] = mapped
    return mapping


def canonical_state(board: np.ndarray):
    """Reduce a board to a canonical key shared by all 8 of its symmetries.

    A 2048 board has the symmetries of a square (4 rotations x optional mirror,
    the dihedral group D4). A rotated or mirrored board is the *same* game
    position, as long as the actions are rotated or mirrored along with it. By
    storing all 8 symmetric variants of a board under a single Q-table key, the
    effective state space shrinks by up to 8x, and experience gathered on one
    variant transfers to the other 7 for free.

    The canonical variant is the one whose tobytes() is lexicographically
    smallest — an arbitrary but consistent choice, which is all a dictionary
    key needs.

    Args:
        board: The raw board observation.

    Returns:
        A (key, action_map) tuple. `key` is the canonical variant's bytes, for
        use as the Q-table key. `action_map` maps each original action index to
        the action index with the same effect on the canonical variant —
        Q(s, a) must be read and written at q_table[key][action_map[a]], never
        at q_table[key][a], or the four actions get scrambled across symmetries.
    """
    best_key = None
    best_map = None
    for num_rotations in range(4):
        rotated = np.rot90(board, num_rotations)
        for flipped in (False, True):
            variant = np.fliplr(rotated) if flipped else rotated
            key = variant.tobytes()
            if best_key is None or key < best_key:
                best_key = key
                best_map = _action_map(num_rotations, flipped)
    return best_key, best_map

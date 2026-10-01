"""Shared per-game seeding for reproducible runs.

The environment's tile spawns come from the global random module, so seeding
it immediately before a game makes that game reproducible on its own, without
replaying the games before it.
"""

import random


def seed_game(run_seed: int, game_number: int) -> int:
    """Seed the global RNG for one game; returns the seed used."""
    seed = run_seed * 1_000_003 + game_number
    random.seed(seed)
    return seed

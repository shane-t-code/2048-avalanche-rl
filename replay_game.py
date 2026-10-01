"""Replays one game in the GUI from its run seed and game number.

Runnable:  python replay_game.py

Any game from rl_play.py or harvest.py can be replayed here by copying its
run_seed and game_number: the same seed reproduces the same game exactly, so
the replay is verifiable by anyone with the checkpoint.
"""

import os

import pygame

from envs.SGAI_2048 import SGAI_2048
from game_seed import seed_game
from models.NTupleAgentInference import NTupleAgentInference

# settings
RUN_SEED = 20260727
GAME_NUMBER = 3900
REPLAY_FPS = 240  # the environment renders one frame per move
MAX_TURNS = 50_000
MODEL_PATH = os.path.join(
    "models", "ntuple_files", "ntuple_6cell_10shapes_1510193.npz"
)

agent = NTupleAgentInference(
    model_file_path=MODEL_PATH,
    search_depth=3,
    deepen_empty_threshold=4,
    deepen_extra=1,
)

env = SGAI_2048(render_mode="human")
env.metadata = {**env.metadata, "render_fps": REPLAY_FPS}

print(f"replaying run_seed={RUN_SEED} game_number={GAME_NUMBER}")
seed_game(RUN_SEED, GAME_NUMBER)
obs_, _ = env.reset()
for turn in range(MAX_TURNS):
    action = agent.get_action(obs_raw=obs_)
    obs_, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

print(f"finished: highest tile {env.highest_tile}, "
      f"score {env.total_score:,} in {turn + 1:,} moves")

# redraw the game-over overlay, which the final step's render pass covers
env._is_game_over()
pygame.display.update()

print("window is open - screenshot it, then press any key or close it")
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type in (pygame.QUIT, pygame.KEYDOWN):
            waiting = False
    pygame.time.wait(50)
env.close()

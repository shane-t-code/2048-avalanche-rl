import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import pygame
from envs.SGAI_2048 import SGAI_2048, Act2048, logger

logger.setLevel(logging.INFO)

# Create the env directly from the class without registering it with gym
env = SGAI_2048(render_mode="human", num_rows=4, num_cols=4)

# Things to keep track of, though there will only be one game at a time
turn_rewards: List[int] = []
overall_score: Union[int, None] = None
all_actions: Dict[Act2048, int] = {
    Act2048.UP: 0,
    Act2048.DOWN: 0,
    Act2048.LEFT: 0,
    Act2048.RIGHT: 0,
}

# Initialize env and counters for human game
obs_, info_ = env.reset()
run_: bool = True
turn_: int = 0

# Start running the game until the game is done
while run_:

    action: Union[None, Act2048] = None
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_LEFT, pygame.K_a]:
                action = Act2048.LEFT
            if event.key in [pygame.K_RIGHT, pygame.K_d]:
                action = Act2048.RIGHT
            if event.key in [pygame.K_UP, pygame.K_w]:
                action = Act2048.UP
            if event.key in [pygame.K_DOWN, pygame.K_s]:
                action = Act2048.DOWN
            if event.key in [pygame.K_q, pygame.K_ESCAPE, pygame.K_BACKSPACE]:
                action = None
                overall_score = env.total_score
                run_ = False
                break

    if action is None:
        continue

    all_actions[Act2048(action)] += 1
    obs_, reward, terminated, truncated, info_ = env.step(action.value)
    turn_rewards.append(reward)
    print(
        f"Turn: {turn_}  -  step reward: {reward}  -  total reward: {env.total_score}"
    )

    if terminated or truncated:
        run_ = False
        overall_score = env.total_score

    turn_ += 1

env.close()

# Plotting
if overall_score is not None:
    plt.bar(["Total Score"], [overall_score], color="blue")
    plt.ylabel("Score (merge points)")
    plt.title("Final Score")
    plt.show()

plt.plot(turn_rewards)
plt.xlabel("Turn")
plt.ylabel("Step Reward")
plt.title("Reward per Turn")
plt.show()

plt.bar(range(len(all_actions)), list(all_actions.values()), align="center")
plt.xticks(range(len(all_actions)), [str(act_) for act_ in list(all_actions.keys())])
plt.xlabel("Action")
plt.ylabel("Times Chosen")
plt.title("Action Counts")
plt.show()

print(f"Score: {overall_score}  -  Highest tile: {env.highest_tile}")

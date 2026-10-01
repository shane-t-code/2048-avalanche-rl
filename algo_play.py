import logging
from typing import Dict, List

import matplotlib.pyplot as plt

from algos.BadDLOnlyAgent import BadDLOnlyAgent
from envs.SGAI_2048 import SGAI_2048, Act2048, logger

logger.setLevel(logging.INFO)

# Create the env directly from the class without registering it with gym
env = SGAI_2048(render_mode="rl-headless", num_rows=4, num_cols=4)

# Things to keep track of, though there will only be one game at a time
turn_rewards: List[int] = []
game_scores: List[int] = []
game_highest_tiles: List[int] = []
all_actions: Dict[Act2048, int] = {
    Act2048.UP: 0,
    Act2048.DOWN: 0,
    Act2048.LEFT: 0,
    Act2048.RIGHT: 0,
}

# Set up runs
num_games = 500
max_turns_per_game = 10_000

# Create Agents to play
# algo_agent = RandomAgent()
algo_agent = BadDLOnlyAgent()

for game_number in range(num_games):

    # Reset for each game
    obs_, info_ = env.reset()
    print(f"**** Game {game_number} of {num_games} ****")

    for turn_number in range(max_turns_per_game):

        action = algo_agent.get_action()

        all_actions[Act2048(action)] += 1
        obs_, reward, terminated, truncated, info_ = env.step(action)
        turn_rewards.append(reward)

        if terminated or truncated:
            break

    game_scores.append(env.total_score)
    game_highest_tiles.append(env.highest_tile)

env.close()

# Plotting
plt.plot(game_scores)
plt.xlabel("Game")
plt.ylabel("Final Score (merge points)")
plt.title("Score per Game")
plt.show()

plt.plot(turn_rewards)
plt.xlabel("Turn (across all games)")
plt.ylabel("Step Reward")
plt.title("Reward per Turn")
plt.show()

plt.bar(range(len(all_actions)), list(all_actions.values()), align="center")
plt.xticks(range(len(all_actions)), [str(act_) for act_ in list(all_actions.keys())])
plt.xlabel("Action")
plt.ylabel("Times Chosen")
plt.title("Action Counts")
plt.show()

print(f"Average score: {sum(game_scores) / len(game_scores)}")
print(f"Max score: {max(game_scores)}")
print(f"Highest tile reached: {max(game_highest_tiles)}")

import logging
import os
import matplotlib.pyplot as plt
from typing import List, Dict
from envs.SGAI_2048 import SGAI_2048, Act2048, logger
from game_seed import seed_game
from models.QAgentInference import QAgentInference
from models.NTupleAgentInference import NTupleAgentInference

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
max_turns_per_game = 50_000  # safety valve only; games end on their own well before this
# each game is seeded from this, so any game in the run can be replayed on its
# own and the whole run reproduces exactly
run_seed = 20260727

# Create Agents to play
# model_weights_path = os.path.join("models", "q_table_files", "q_table_final.pickle")
# rl_agent = QAgentInference(q_table_file_path=model_weights_path)
# model_weights_path = os.path.join('models', 'dq_model_files', 'dq_policy_final.pth')
# rl_agent = DQAgentInference(model_file_path=model_weights_path)
model_weights_path = os.path.join(
    "models", "ntuple_files", "ntuple_6cell_10shapes_1510193.npz"
)
rl_agent = NTupleAgentInference(
    model_file_path=model_weights_path,
    search_depth=2,
    deepen_empty_threshold=4,
    deepen_extra=1,
)

for game_number in range(num_games):

    # Reset for each game
    seed_game(run_seed, game_number)
    obs_, info_ = env.reset()
    print(f"**** Game {game_number} of {num_games} ****")

    for turn_number in range(max_turns_per_game):

        action = rl_agent.get_action(obs_raw=obs_)

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

# The two lines above are maxima over all games and can come from different
# games, so also report the single best game (highest tile first, then score).
best_game_number, (best_tile, best_score) = max(
    enumerate(zip(game_highest_tiles, game_scores)), key=lambda pair: pair[1]
)
print(f"Best single game: #{best_game_number} - highest tile {best_tile}, "
      f"score {best_score}")
print(f"Replay it with: run_seed={run_seed}, game_number={best_game_number}")

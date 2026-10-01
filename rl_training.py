import logging
import os
import pickle
from typing import Dict, List

import gymnasium
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from envs.SGAI_2048 import SGAI_2048, Act2048, logger
from models.DQAgentTrainer import DQAgentTrainer
from models.QAgentTrainer import QAgentTrainer

logger.setLevel(logging.INFO)

# Create the env directly from the class without registering it with gym
env = SGAI_2048(render_mode="rl-headless", num_rows=4, num_cols=4)

# Things to keep track of, though there will only be one game at a time
turn_rewards: List[int] = []
game_scores: List[int] = []
game_highest_tiles: List[int] = []
policy_loss_by_game: Dict[int, List[float]] = dict()
all_actions: Dict[Act2048, int] = {
    Act2048.UP: 0,
    Act2048.DOWN: 0,
    Act2048.LEFT: 0,
    Act2048.RIGHT: 0,
}

# Set up runs
max_turns_per_game = 10_000
# model_output = os.path.join("models", "q_table_files")
model_output = os.path.join("models", "dq_model_files")

# hyperparameters
learning_rate = 0.0001  # 0.001 (for q learning)
n_episodes = 5_000  # 1_000_000_000
start_epsilon = 0.9
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

# rl_agent = QAgentTrainer(
#     learning_rate=learning_rate,
#     initial_epsilon=start_epsilon,
#     epsilon_decay=epsilon_decay,
#     final_epsilon=final_epsilon,
# )

rl_agent = DQAgentTrainer(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

env = gymnasium.wrappers.RecordEpisodeStatistics(env)

for game_number in tqdm(range(n_episodes)):

    # Reset for each game
    obs_, info_ = env.reset()
    policy_losses: List[float] = []

    for turn_number in range(max_turns_per_game):

        action = rl_agent.get_action(obs_raw=obs_)
        all_actions[Act2048(action)] += 1
        obs_new, reward, terminated, truncated, info_ = env.step(action)
        turn_rewards.append(int(reward))

        # update the agent
        policy_loss = rl_agent.update(
            obs_raw=obs_,
            action=action,
            reward=reward,
            terminated=(terminated or truncated),
            next_obs_raw=obs_new,
        )

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs_ = obs_new

        policy_losses.append(policy_loss)

        if done:
            break

    rl_agent.decay_epsilon()
    game_scores.append(env.get_wrapper_attr("total_score"))
    game_highest_tiles.append(env.get_wrapper_attr("highest_tile"))
    policy_loss_by_game[game_number] = policy_losses

    # write model out every X games
    if game_number > 0 and game_number % 10_000 == 0:
        torch.save(
            rl_agent.policy.state_dict(),
            os.path.join(model_output, f"dq_policy_{game_number}.pth"),
        )
        # with open(
        #     os.path.join(model_output, f"q_table_{game_number}.pickle"), "wb"
        # ) as f_:
        #     pickle.dump(dict(rl_agent.q_values), f_, protocol=pickle.HIGHEST_PROTOCOL)

# write out final model
torch.save(
    rl_agent.policy.state_dict(), os.path.join(model_output, "dq_policy_final.pth")
)
# with open(os.path.join(model_output, "q_table_final.pickle"), "wb") as f_:
#     pickle.dump(dict(rl_agent.q_values), f_, protocol=pickle.HIGHEST_PROTOCOL)

env.close()

# Plotting
plt.plot(game_scores)
plt.xlabel("Episode")
plt.ylabel("Final Score (merge points)")
plt.title("Score per Training Episode")
plt.show()

plt.plot(turn_rewards)
plt.xlabel("Turn (across all episodes)")
plt.ylabel("Step Reward")
plt.title("Reward per Turn")
plt.show()

plt.bar(range(len(all_actions)), list(all_actions.values()), align="center")
plt.xticks(range(len(all_actions)), [str(act_) for act_ in list(all_actions.keys())])
plt.xlabel("Action")
plt.ylabel("Times Chosen")
plt.title("Action Counts")
plt.show()

plt.plot(rl_agent.training_error)
plt.xlabel("Update Step")
plt.ylabel("Training Error (TD error / loss)")
plt.title("Training Error over Updates")
plt.show()

max_score = max(game_scores)
max_game_number = game_scores.index(max_score)
print(f"Max Score: {max_score} on game number {max_game_number}")
print(f"Highest tile reached: {max(game_highest_tiles)}")

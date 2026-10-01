import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from envs.SGAI_2048 import SGAI_2048, Act2048


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        """Initialize Q function with 2 hidden layers

        Args:
            n_observations: dimensions of the observation
            n_actions: dimensions of the possible actions
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQAgentTrainer:

    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        batch_size: float = 1,
        discount_factor: float = 0.99,
        device: str = "cpu",
    ):
        """
        Initialize a Reinforcement Learning agent with a behavioral policy and a target policy,
        and batch_size for the NNs, as well as both the learning rate and epsilon seen before.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        example_env = SGAI_2048()
        obs_, _ = example_env.reset()
        state_shape_flattened = len(obs_.flatten())
        num_actions = len(Act2048)
        self.policy = DQN(state_shape_flattened, num_actions).to(device).train()
        self.target = DQN(state_shape_flattened, num_actions).to(device).eval()
        self.target.load_state_dict(self.policy.state_dict())
        self.device = device
        self.memory = []
        self.buffer_size = 100000
        self.optimizer = optim.AdamW(
            self.policy.parameters(), lr=learning_rate, amsgrad=True
        )
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.action_list = [act_.value for act_ in Act2048]

    def get_action(self, obs_raw):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            action: int = random.choice(self.action_list)
            return action
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            with torch.no_grad():
                return (
                    self.policy(torch.tensor(np.float32(obs_raw.flatten())))
                    .max(0)
                    .indices.item()
                )

    def update(
        self, obs_raw, action: int, reward: float, terminated: bool, next_obs_raw
    ):
        self.memory.append(
            (obs_raw.flatten(), action, next_obs_raw.flatten(), reward, terminated)
        )
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size :]

        policy_loss = self.optimize()
        self.training_error.append(policy_loss)
        self.update_target_policy()

        return policy_loss

    def optimize(self):

        if len(self.memory) < self.batch_size:
            return np.inf

        transitions = random.sample(self.memory, self.batch_size)

        # get batches of tensors
        state = torch.tensor(np.array([x[0] for x in transitions]))
        next_state = torch.tensor(np.array([x[2] for x in transitions]))
        action = torch.tensor([x[1] for x in transitions]).unsqueeze(
            1
        )  # match dimensions
        reward = torch.tensor([x[3] for x in transitions]).unsqueeze(
            1
        )  # match dimensions
        terminated = torch.tensor(
            [x[4] for x in transitions], dtype=torch.float32
        ).unsqueeze(1)

        # calculate expected q-values from current state
        expected_reward_now = self.policy(state.float()).gather(1, action)

        # calculate expected q-values from next state, zero for terminal transitions
        with torch.no_grad():
            next_state_values = (
                self.target(next_state.float()).max(1).values.unsqueeze(1)
            )
        expected_reward_next = (
            next_state_values * self.discount_factor * (1.0 - terminated)
        ) + reward

        # Compute Huber loss over difference between expected values
        criterion = nn.SmoothL1Loss()
        loss = criterion(expected_reward_now, expected_reward_next)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 1)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def update_target_policy(self, tau=0.005):
        target_state_dict = self.target.state_dict()
        policy_state_dict = self.policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * tau + target_state_dict[
                key
            ] * (1 - tau)
        self.target.load_state_dict(target_state_dict)

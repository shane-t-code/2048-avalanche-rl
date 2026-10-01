from collections import defaultdict
import numpy as np
import random
from envs.SGAI_2048 import Act2048
from utils.state_keys import canonical_state


class QAgentTrainer:

    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        seed=None,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        # TODO HW: estimate (even roughly) how many distinct 4x4 boards this game
        # can reach. Can this dictionary ever hold them all? What does your answer
        # say about what a tabular agent has actually learned when training ends?
        # The canonical_state symmetry trick below buys an 8x reduction — does
        # that change your conclusion?
        self.q_values = defaultdict(lambda: np.zeros(len(Act2048)))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.action_list = [act_.value for act_ in Act2048]

    def get_action(self, obs_raw: np.ndarray) -> int:
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
            # all 8 symmetric variants of a board share one Q-table entry; the
            # action map translates each of our actions into that entry's frame
            state_key, action_map = canonical_state(obs_raw)
            q_row = self.q_values[state_key]
            return int(np.argmax([q_row[action_map[a]] for a in self.action_list]))

    def update(
        self,
        obs_raw: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_obs_raw: np.ndarray,
    ):
        """Updates the Q-value of an action."""

        # canonical keys so all 8 symmetric variants train the same entry; the
        # taken action must be translated into the canonical frame too. The max
        # over the next state's Q-row needs no translation — a max is the same
        # under any reordering of the actions
        state_key_current, action_map_current = canonical_state(obs_raw)
        state_key_next, _ = canonical_state(next_obs_raw)
        canonical_action = action_map_current[action]

        future_q_value = (not terminated) * np.max(self.q_values[state_key_next])
        temporal_difference = (
            reward
            + self.discount_factor * future_q_value
            - self.q_values[state_key_current][canonical_action]
        )

        self.q_values[state_key_current][canonical_action] = (
            self.q_values[state_key_current][canonical_action]
            + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

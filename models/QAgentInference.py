import numpy as np
import random
from envs.SGAI_2048 import Act2048
from utils.state_keys import canonical_state
import pickle


class QAgentInference:

    def __init__(self, q_table_file_path: str):
        with open(q_table_file_path, "rb") as handle:
            self.q_values = pickle.load(handle)
        self.action_list = [act_.value for act_ in Act2048]

    def get_action(self, obs_raw: np.ndarray) -> int:
        # same canonical key + action translation the trainer used, so lookups
        # hit the entries training actually wrote
        state_key, action_map = canonical_state(obs_raw)
        # plain .get() so unseen states don't silently grow the table
        q_row = self.q_values.get(state_key)
        if q_row is None or not np.any(q_row):
            # a state the training run never saw (or learned nothing about):
            # argmax of an all-zero row would always return UP, so fall back to
            # a random move and degrade to RandomAgent instead of always-UP
            return random.choice(self.action_list)
        return int(np.argmax([q_row[action_map[a]] for a in self.action_list]))

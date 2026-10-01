import random
from envs.SGAI_2048 import Act2048


class RandomAgent:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.actions = [act_.value for act_ in Act2048]

    def get_action(self):
        random_action = random.choice(self.actions)
        return random_action

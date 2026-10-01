import numpy as np
import torch
import torch.nn as nn
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


class DQAgentInference:

    def __init__(self, model_file_path: str, device: str = "cpu"):
        self.device = device
        example_env = SGAI_2048()
        obs_, _ = example_env.reset()
        state_shape_flattened = len(obs_.flatten())
        num_actions = len(Act2048)
        self.policy = DQN(state_shape_flattened, num_actions).to(device)
        self.policy.load_state_dict(torch.load(model_file_path))
        self.policy.eval()

    def get_action(self, obs_raw):
        with torch.no_grad():
            return (
                self.policy(torch.tensor(np.float32(obs_raw.flatten())))
                .max(0)
                .indices.item()
            )

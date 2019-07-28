"""
Soft actor critic per "Soft Actor-Critic Algorithms and Applications" Haarnoja
et. al 2019.
"""
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn

sys.path.insert(0, '..')
from neural_nets.base_network import RectangleNN
from rlmethods.rlutils import ReplayBuffer


class QNetwork(RectangleNN):
    def __init__(self, num_layers, layer_width):
        super().__init__(num_layers, layer_width, F.relu)

    def forward(self, x):
        pass


class PolicyNetwork(RectangleNN):
    """Policy network for soft actor critic."""

    def __init__(self, num_layers, hidden_layer_width, out_layer_width):
        super().__init__(num_layers, hidden_layer_width, F.relu)

        self.head = nn.Linear(hidden_layer_width, out_layer_width)

    def forward(self, x):
        x = self.hidden_forward(x)
        x = F.softmax(self.head(x), dim=-1)

        return x

    def action_distribution(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        return dist


class SoftActorCritic:
    def __init__(self, env, replay_buffer_size=10**6):
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # NNs
        self.policy = PolicyNetwork(2, 512, 4)

    def select_action(self, state):
        dist = self.policy.action_distribution(state)
        action = dist.sample()

        return action

    def populate_samples(self):
        state = self.env.reset()

        while not self.replay_buffer.is_full():
            action = self.select_action(state)
            next_state, reward, done = self.env.step(action)

"""
Soft actor critic per "Soft Actor-Critic Algorithms and Applications" Haarnoja
et. al 2019.
"""
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn

sys.path.insert(0, '..')
from rlmethods.rlutils import ReplayBuffer  # NOQA
from neural_nets.base_network import RectangleNN  # NOQA

DEVICE = ('gpu' if torch.cuda.is_available() else 'cpu')


def copy_params(source, target):
    """Copies parameters from source network to target network.

    :param source: Network to copy parameters from.
    :param target: Network to copy parameters to.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def move_average(source, target, tau=0.005):
    """Compute and set moving average for target network.

    :param source: Network with new weights.
    :param target: Network to whose weights are moved closer to source.
    :param tau: Average rate, default 0.005 as per paper.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class QNetwork(RectangleNN):
    """Q function network."""

    def __init__(self, num_inputs, num_layers, hidden_layer_width):
        super().__init__(num_layers, hidden_layer_width, F.relu)

        self.in_layer = nn.Linear(num_inputs, hidden_layer_width)
        self.head = nn.Linear(hidden_layer_width, 1)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = self.hidden_forward(x)
        x = self.head(x)

        return x


class PolicyNetwork(RectangleNN):
    """Policy network for soft actor critic."""

    def __init__(
            self,
            num_layers,
            num_inputs,
            hidden_layer_width,
            out_layer_width
    ):
        super().__init__(num_layers, hidden_layer_width, F.relu)

        self.in_layer = nn.Linear(num_inputs, hidden_layer_width)
        self.head = nn.Linear(hidden_layer_width, out_layer_width)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = self.hidden_forward(x)
        x = F.softmax(self.head(x), dim=-1)

        return x

    def action_distribution(self, state):
        """Returns a pytorch distribution object based on policy output.

        :param state: Input state vector.
        """
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        return dist


class SoftActorCritic:
    """Implementation of soft actor critic."""

    def __init__(self, env, replay_buffer_size=10**6):
        self.env = env
        starting_state = self.env.reset()
        state_size = starting_state.shape[0]
        action_size = env.action_space.n

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # NNs
        self.policy = PolicyNetwork(2, state_size, 512, action_size).to(DEVICE)
        self.q_net = QNetwork(state_size + action_size, 2, 512).to(DEVICE)
        self.avg_q_net = QNetwork(state_size + action_size, 2, 512).to(DEVICE)

        # initialize weights of moving avg Q net
        copy_params(self.q_net, self.avg_q_net)

        # initialize temperature
        self.alpha = torch.tensor([1]).to(DEVICE)

    def select_action(self, state):
        """Generate an action based on state vector using current policy.

        :param state: Current state vector.
        """
        dist = self.policy.action_distribution(state)
        action = dist.sample()

        return action

    def populate_buffer(self):
        """
        Fill in entire replay buffer with state action pairs using current
        policy.
        """
        state = self.env.reset()

        while not self.replay_buffer.is_full():
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push((state, action, reward, next_state, done))

    def train(self):
        """Train Soft Actor Critic"""
        # Populate the buffer
        self.populate_buffer()

        # weight updates
        replay_samples = self.replay_buffer.sample(100)
        states = torch.from_numpy(replay_samples[0]).to(DEVICE)
        actions = torch.from_numpy(replay_samples[1]).to(DEVICE)
        rewards = torch.from_numpy(replay_samples[2]).to(DEVICE)
        next_states = torch.from_numpy(replay_samples[3]).to(DEVICE)
        dones = torch.tensor(replay_samples[4]).to(DEVICE)

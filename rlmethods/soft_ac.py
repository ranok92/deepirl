"""
Soft actor critic per "Soft Actor-Critic Algorithms and Applications" Haarnoja
et. al 2019.
"""
import sys
import pdb

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

from tensorboardX import SummaryWriter

sys.path.insert(0, '..')
from rlmethods.rlutils import ReplayBuffer  # NOQA
from neural_nets.base_network import RectangleNN  # NOQA

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')


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


def to_oh(index, num_classes):
    """Convert to one hot.

    :param index: 'hot' index (which index to set to one)
    :param num_classes: total number of classes (length of vector)
    """
    oh = np.zeros(num_classes).astype('float32')
    oh[index] = 1

    return oh


class QNetwork(RectangleNN):
    """Q function network."""

    def __init__(self, state_length, action_length, num_layers, hidden_layer_width):
        super().__init__(num_layers, hidden_layer_width, F.relu)

        self.action_length = action_length

        self.in_layer = nn.Linear(
            state_length+action_length,
            hidden_layer_width
        )
        self.head = nn.Linear(hidden_layer_width, 1)

    def forward(self, states, actions):
        actions_vector = torch.eye(self.action_length)[actions]
        x = torch.cat([states, actions_vector], 1)
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
        x = self.head(x)
        probs = F.softmax(x, dim=-1)
        log_probs = F.log_softmax(x, dim=-1)

        return probs, log_probs

    def action_distribution(self, state):
        """Returns a pytorch distribution object based on policy output.

        :param state: Input state vector.
        """
        probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        return dist


class SoftActorCritic:
    """Implementation of soft actor critic."""

    def __init__(
            self,
            env,
            replay_buffer_size=10**6,
            buffer_sample_size=10**4,
            gamma=1.0,
            learning_rate=3 * 10**-4,
            tbx_writer = None,
    ):
        self.env = env
        starting_state = self.env.reset()
        state_size = starting_state.shape[0]
        action_size = env.action_space.n

        # buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.buffer_sample_size = buffer_sample_size

        # NNs
        self.policy = PolicyNetwork(2, state_size, 512, action_size).to(DEVICE)
        self.q_net = QNetwork(state_size, action_size, 2, 512).to(DEVICE)
        self.avg_q_net = QNetwork(state_size, action_size, 2, 512).to(DEVICE)

        # initialize weights of moving avg Q net
        copy_params(self.q_net, self.avg_q_net)

        # set hyperparameters
        self.alpha = torch.tensor([0.0], requires_grad=True).to(DEVICE)
        self.gamma = gamma
        self.entropy_target = -action_size

        # training meta
        self.training_i = 0

        # optimizers
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)
        self.q_optim = Adam(self.q_net.parameters(), lr=learning_rate)
        self.alpha_optim = Adam([self.alpha], lr=learning_rate)

        # tensorboardX settings
        if not tbx_writer:
            self.tbx_writer = SummaryWriter('runs/generic_soft_ac')
        else:
            self.tbx_writer = tbx_writer

    def select_action(self, state):
        """Generate an action based on state vector using current policy.

        :param state: Current state vector.
        """
        dist = self.policy.action_distribution(state)
        action = dist.sample()

        return action, dist.log_prob(action)

    def populate_buffer(self):
        """
        Fill in entire replay buffer with state action pairs using current
        policy.
        """
        state = self.env.reset()
        current_state = state

        while not self.replay_buffer.is_full():
            action, _ = self.select_action(torch.from_numpy(state).to(DEVICE))
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push((
                current_state,
                action.cpu().numpy(),
                reward,
                next_state,
                done
            ))
            if not done:
                current_state = next_state
            else:
                current_state = self.env.reset()

    def train(self):
        """Train Soft Actor Critic"""
        # Populate the buffer
        self.populate_buffer()

        # weight updates
        # TODO: which of the below are actually needed?
        replay_samples = self.replay_buffer.sample(100)
        state_batch = torch.from_numpy(replay_samples[0]).to(DEVICE)
        action_batch = torch.from_numpy(replay_samples[1]).to(DEVICE)
        reward_batch = torch.from_numpy(replay_samples[2]).to(DEVICE)
        next_state_batch = torch.from_numpy(replay_samples[3]).to(DEVICE)
        # dones_batch = torch.tensor(replay_samples[4]).to(DEVICE)

        with torch.no_grad():
            # Figure out value function
            next_actions, log_next_actions = self.select_action(
                next_state_batch
            )
            next_state_q = self.avg_q_net(next_state_batch, next_actions)
            next_state_values = next_state_q - self.alpha * log_next_actions

            # Calculate Q network target
            q_net_target = reward_batch + self.gamma * next_state_values

        # q network loss
        q_values = self.q_net(state_batch, action_batch)
        q_loss = F.mse_loss(q_values, q_net_target)

        self.tbx_writer.add_scalar('Q loss', q_loss.item(), self.training_i)

        # policy loss
        _, log_actions = self.select_action(state_batch)
        policy_loss = (self.alpha * log_actions - q_values.detach()).mean()

        self.tbx_writer.add_scalar(
            'pi loss',
            policy_loss.item(),
            self.training_i
        )

        # automatic entropy tuning
        alpha_loss = -self.alpha * (log_actions + self.entropy_target).detach()
        alpha_loss = alpha_loss.mean()

        self.tbx_writer.add_scalar(
            'alpha loss',
            alpha_loss.item(),
            self.training_i
        )

        self.tbx_writer.add_scalar('alpha', self.alpha.item(), self.training_i)

        # update parameters
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Step average Q net
        move_average(self.q_net, self.avg_q_net)

        self.training_i += 1

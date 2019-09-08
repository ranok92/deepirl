"""
Soft actor critic per "Soft Actor-Critic Algorithms and Applications" Haarnoja
et. al 2019.
"""
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical

from tensorboardX import SummaryWriter

sys.path.insert(0, '..')
from rlmethods.rlutils import ReplayBuffer  # NOQA
from neural_nets.base_network import BaseNN  # NOQA

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

def get_action_q(all_q_a, action_indices):
    """Q network generates all Q values for all possible actions. Use this
    function to get the Q value of a single specific action, whose index is
    specific in action_indices. This works with batches.

    :param all_q_a: NxA tensor of Q values for N rows of length A when A is
    number of actions possible.
    :param action_indices: 1xN tensor of shape (N,) of action indices.
    """
    index_tuple = (
        torch.arange(len(action_indices)),
        action_indices.type(torch.long)
    )

    return all_q_a[index_tuple]


class QNetwork(BaseNN):
    """Q function network."""

    def __init__(self, state_length, action_length, hidden_layer_width):
        super().__init__()

        self.action_length = action_length

        self.in_layer = nn.Linear(state_length, hidden_layer_width)
        self.hidden1 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.hidden2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.head = nn.Linear(hidden_layer_width, action_length)

    def forward(self, states):
        # actions need to be byte or long to be used as indices
        x = F.relu(self.in_layer(states))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.head(x)

        return x


class PolicyNetwork(BaseNN):
    """Policy network for soft actor critic."""

    def __init__(
            self,
            num_inputs,
            hidden_layer_width,
            out_layer_width
    ):
        super().__init__()

        self.in_layer = nn.Linear(num_inputs, hidden_layer_width)
        self.hidden1 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.hidden2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.head = nn.Linear(hidden_layer_width, out_layer_width)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.head(x)
        probs = F.softmax(x, dim=-1)

        return probs

    def action_distribution(self, state):
        """Returns a pytorch distribution object based on policy output.

        :param state: Input state vector.
        """
        probs = self.__call__(state)
        dist = Categorical(probs)

        return dist



class SoftActorCritic:
    """Implementation of soft actor critic."""

    def __init__(
            self,
            env,
            replay_buffer_size=10**6,
            buffer_sample_size=10**4,
            gamma=0.99,
            learning_rate=3 * 10**-4,
            tbx_writer=None,
            entropy_tuning=False,
            tau=0.005,
            log_alpha=-2.995,
    ):
        self.env = env
        starting_state = self.env.reset()
        state_size = starting_state.shape[0]

        # buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.buffer_sample_size = buffer_sample_size

        # NNs
        self.policy = PolicyNetwork(state_size, 256, env.action_space.n)
        self.q_net = QNetwork(state_size, env.action_space.n, 256)
        self.avg_q_net = QNetwork(state_size, env.action_space.n, 256)

        # initialize weights of moving avg Q net
        copy_params(self.q_net, self.avg_q_net)

        # set hyperparameters
        self.log_alpha = torch.tensor([log_alpha], requires_grad=True).to(DEVICE)
        self.gamma = gamma
        self.entropy_target = -1
        self.tau = tau

        # training meta
        self.training_i = 0
        self.entropy_tuning = entropy_tuning

        # optimizers
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)
        self.q_optim = Adam(self.q_net.parameters(), lr=learning_rate)
        self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)

        # tensorboardX settings
        if not tbx_writer:
            self.tbx_writer = SummaryWriter('runs/generic_soft_ac')
        else:
            self.tbx_writer = tbx_writer

    def select_action(self, state):
        """Generate an action based on state vector using current policy.

        :param state: Current state vector. must be Torch 32 bit float tensor.
        """
        dist = self.policy.action_distribution(state)
        action = dist.sample()

        return action, dist.log_prob(action), dist

    def populate_buffer(self):
        """
        Fill in entire replay buffer with state action pairs using current
        policy.
        """
        done = True

        while not self.replay_buffer.is_full():
            if done:
                current_state = self.env.reset()
                done = False

            while not done:
                _state = torch.from_numpy(current_state).type(
                    torch.float).to(DEVICE)
                action, _, _ = self.select_action(_state)
                next_state, reward, done, _ = self.env.step(action.item())
                self.replay_buffer.push((
                    current_state,
                    action.cpu().numpy(),
                    reward,
                    next_state,
                    not done
                ))
                current_state = next_state


    def tbx_logger(self, log_dict, training_i):
        """Logs the tag-value pairs in log_dict using TensorboardX.

        :param log_dict: {tag:value} dictionary to log.
        :param training_i: Current training iteration.
        """
        for tag, value in log_dict.items():
            self.tbx_writer.add_scalar(tag, value, training_i)

    def train(self):
        """Train Soft Actor Critic"""

        # Populate the buffer
        self.populate_buffer()

        # weight updates
        replay_samples = self.replay_buffer.sample(self.buffer_sample_size)
        state_batch = torch.from_numpy(replay_samples[0]).to(DEVICE)
        action_batch = torch.from_numpy(replay_samples[1]).to(DEVICE)
        reward_batch = torch.from_numpy(replay_samples[2]).to(DEVICE)
        next_state_batch = torch.from_numpy(replay_samples[3]).to(DEVICE)
        dones = torch.from_numpy(replay_samples[4]).type(torch.long).to(DEVICE)

        # alpha must be clamped with a minumum of zero, so use exponential.
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            # Figure out value function
            next_actions, log_next_actions, _ = self.select_action(
                next_state_batch
            )
            next_q_a = self.avg_q_net(next_state_batch)
            next_q = get_action_q(next_q_a, next_actions)
            next_state_values = next_q - alpha * log_next_actions

            # Calculate Q network target
            done_floats = dones.type(torch.float)
            q_target = reward_batch.clone()
            q_target += self.gamma * done_floats * next_state_values.squeeze()

        # q network loss
        q_a_values = self.q_net(state_batch)

        # Q net outputs values for all actions, so we index specific actions
        # TODO: It should be possible to just do MSE over all q_a pairs.
        q_values = get_action_q(q_a_values, action_batch)
        q_loss = F.mse_loss(q_values, q_target)

        # policy loss
        _, log_actions, action_dist = self.select_action(state_batch)
        q_a_pi = self.q_net(state_batch)
        q_dist = Categorical(F.softmax((1.0/alpha)*q_a_pi, dim=-1))

        # compute KL-divergence directly and manually for sanity
        action_probs = action_dist.probs
        q_probs = q_dist.probs
        policy_loss = (action_probs * (action_probs / q_probs).log())
        policy_loss = policy_loss.sum(dim=1).mean()

        # update parameters
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # automatic entropy tuning
        alpha_loss = self.log_alpha * \
            (log_actions + self.entropy_target).detach()
        alpha_loss = -alpha_loss.mean()

        if self.entropy_tuning:
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        # Step average Q net
        move_average(self.q_net, self.avg_q_net, self.tau)


        # logging
        self.tbx_logger(
            {
                'loss/Q loss': q_loss.item(),
                'loss/pi loss': policy_loss.item(),
                'loss/alpha loss': alpha_loss.item(),
                'Q/avg_q_target': q_target.mean().item(),
                'Q/avg_q': q_values.mean().item(),
                'Q/avg_reward': reward_batch.mean().item(),
                'Q/avg_V': next_state_values.mean().item(),
                'Q/entropy': q_dist.entropy().mean(),
                'pi/avg_entropy': action_dist.entropy().mean(),
                'pi/avg_log_actions': log_actions.detach().mean().item(),
                'alpha': alpha.item(),
            },
            self.training_i
        )

        self.training_i += 1
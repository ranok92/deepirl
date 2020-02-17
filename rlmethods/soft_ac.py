"""
Soft actor critic per "Soft Actor-Critic Algorithms and Applications" Haarnoja
et. al 2019.
"""
import sys
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

sys.path.insert(0, '..')
from rlmethods.rlutils import ReplayBuffer  # NOQA
from neural_nets.base_network import BaseNN, reset_parameters  # NOQA

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

MAX_FLOAT = torch.finfo(torch.float32).max
FEPS = torch.finfo(torch.float32).eps


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

        self.alpha=1.0

    def forward(self, states):
        # actions need to be byte or long to be used as indices
        x = F.relu(self.in_layer(states))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.head(x)

        return x

    def sample_action(self, state):
        softmax_over_actions = F.softmax(
            (1.0/self.alpha) * self.__call__(state),
            dim=-1
        )
        dist = Categorical(softmax_over_actions)
        action = dist.sample()

        return action, dist.log_prob(action), dist

    def eval_action(self, state):
        softmax_over_actions = F.softmax(
            (1.0/self.alpha) * self.__call__(state),
            dim=-1
        )
        action = torch.argmax(softmax_over_actions)

        return action


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
            replay_buffer,
            feature_extractor,
            buffer_sample_size=10**4,
            gamma=0.99,
            learning_rate=3e-4,
            tbx_writer=None,
            entropy_tuning=False,
            entropy_target=1.0,
            tau=0.005,
            log_alpha=-2.995,
            q_net=None,
            play_interval=1,
    ):
        self.env = env
        self.feature_extractor = feature_extractor

        starting_state = self.env.reset()

        if self.feature_extractor is not None:
            starting_state = self.feature_extractor.extract_features(starting_state)
        state_size = starting_state.shape[0]

        # buffer
        self.replay_buffer = replay_buffer
        self.buffer_sample_size = buffer_sample_size

        # NNs
        if not q_net:
            self.q_net = QNetwork(state_size, env.action_space.n, 256)
        else:
            self.q_net = q_net

        self.avg_q_net = copy.deepcopy(self.q_net)

        # Dummy policy, it's just q_net in disguise!
        self.policy = self.q_net

        # initialize weights of moving avg Q net
        copy_params(self.q_net, self.avg_q_net)
        self.q_net.to(DEVICE)
        self.avg_q_net.to(DEVICE)

        # set hyperparameters
        self.log_alpha = torch.tensor(log_alpha).to(DEVICE)
        self.log_alpha = self.log_alpha.detach().requires_grad_(True)
        self.gamma = gamma
        self.entropy_target = entropy_target
        self.tau = tau

        # training meta
        self.training_i = 0
        self.play_i = 0
        self.entropy_tuning = entropy_tuning
        self.play_interval = play_interval

        # optimizers
        self.learning_rate = learning_rate
        self.q_optim = Adam(self.q_net.parameters(), lr=learning_rate)
        self.alpha_optim = Adam([self.log_alpha], lr=1e-2)

        # tensorboardX settings
        if not tbx_writer:
            self.tbx_writer = SummaryWriter()
        else:
            self.tbx_writer = tbx_writer

    def select_action(self, state, alpha):
        """Generate an action based on state vector using current policy.

        :param state: Current state vector. must be Torch 32 bit float tensor.
        """
        softmax_over_actions = F.softmax(
            (1.0/alpha) * self.q_net(state),
            dim=-1
        )
        dist = Categorical(softmax_over_actions)
        action = dist.sample()

        return action, dist.log_prob(action), dist

    def populate_buffer(self, max_env_steps):
        """
        Fill in entire replay buffer with state action pairs using current
        policy.
        """
        while len(self.replay_buffer) < self.buffer_sample_size:
            self.play(max_env_steps)

    def reset_training(self):
        self.q_net.apply(reset_parameters)
        self.avg_q_net = copy.deepcopy(self.q_net)
        self.q_optim = Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.alpha_optim = Adam([self.log_alpha], lr=1e-2)

    def tbx_logger(self, log_dict, training_i):
        """Logs the tag-value pairs in log_dict using TensorboardX.

        :param log_dict: {tag:value} dictionary to log.
        :param training_i: Current training iteration.
        """
        for tag, value in log_dict.items():
            self.tbx_writer.add_scalar(tag, value, training_i)

    def train_episode(self, max_env_steps, reward_net=None):
        """Train Soft Actor Critic"""

        # Populate the buffer
        self.populate_buffer(max_env_steps)

        # weight updates
        replay_samples = self.replay_buffer.sample(self.buffer_sample_size)
        state_batch = torch.from_numpy(replay_samples[0]).to(DEVICE)
        action_batch = torch.from_numpy(replay_samples[1]).to(DEVICE)
        reward_batch = torch.from_numpy(replay_samples[2]).to(DEVICE)
        next_state_batch = torch.from_numpy(replay_samples[3]).to(DEVICE)
        dones = torch.from_numpy(replay_samples[4]).type(torch.long).to(DEVICE)

        # if reward net exists, replace reward batch with fresh one
        if reward_net:
            fresh_rewards = reward_net(next_state_batch).flatten()
            assert fresh_rewards.shape == reward_batch.shape
            reward_batch = fresh_rewards

        # alpha must be clamped with a minumum of zero, so use exponential.
        alpha = self.log_alpha.exp().detach()
        self.policy.alpha = alpha

        with torch.no_grad():
            # Figure out value function
            next_actions, log_next_actions, _ = self.select_action(
                next_state_batch,
                alpha
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
        q_values = get_action_q(q_a_values, action_batch.squeeze())
        q_loss = F.mse_loss(q_values, q_target)

        # policy loss
        _, log_actions, action_dist = self.select_action(state_batch, alpha)

        # update parameters
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

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
                'loss/alpha loss': alpha_loss.item(),
                'Q/avg_q_target': q_target.mean().item(),
                'Q/avg_q': q_values.mean().item(),
                'Q/avg_reward': reward_batch.mean().item(),
                'Q/avg_V': next_state_values.mean().item(),
                'pi/avg_entropy': action_dist.entropy().mean(),
                'pi/avg_log_actions': log_actions.detach().mean().item(),
                'alpha': alpha.item(),
            },
            self.training_i
        )

        self.training_i += 1

    def play(self, max_env_steps, reward_network=None, render=False, best_action=False):
        """
        Play one complete episode in the environment's gridworld.
        Automatically appends to replay buffer, and logs with Tensorboardx.

        :param max_env_steps: Maximum number of steps to take in playthrough.
        :param reward_network: Replaces environment's builtin rewards.
        :param render: If True, renders the playthrough.
        :param best_action: If True, uses best actions instead of stochastic actions.
        """

        done = False
        total_reward = np.zeros(1)
        state = self.env.reset()
        if self.feature_extractor is not None:
            state = self.feature_extractor.extract_features(state)
        episode_length = 0

        for _ in range(max_env_steps):
            # Env returns numpy state so convert to torch
            torch_state = torch.from_numpy(state).type(torch.float32)
            torch_state = torch_state.to(DEVICE)

            alpha = self.log_alpha.exp().detach()

            # select an action to do
            if best_action:
                action = self.policy.eval_action(torch_state)
            else:
                action, _, _ = self.select_action(torch_state, alpha)

            next_state, reward, done, _ = self.env.step(action.item())
            next_state = self.feature_extractor.extract_features(next_state)

            if render:
                self.env.render()

            if reward_network:
                reward = reward_network(torch_state).cpu().item()

            if episode_length > max_env_steps:
                self.replay_buffer.push((
                    state,
                    action.cpu().numpy(),
                    reward,
                    next_state,
                    done
                ))
            else:
                self.replay_buffer.push((
                    state,
                    action.cpu().numpy(),
                    reward,
                    next_state,
                    not done
                ))

            state = next_state
            total_reward += reward
            episode_length += 1

            if done:
                break

        self.tbx_writer.add_scalar(
            'rewards/episode_reward',
            total_reward.item(),
            self.play_i
        )

        self.tbx_writer.add_scalar(
            'rewards/episode_length',
            episode_length,
            self.play_i
        )

        self.play_i += 1

    def train(self, num_episodes, max_env_steps, reward_network=None):
        """Train and play environment every play_interval, appending obtained
        states, actions, rewards, and dones to the replay buffer.

        :param num_episodes: number of episodes to train for.
        :param play_interval: trainig episodes between each play session.
        """

        print("Training RL . . .")

        for _ in tqdm(range(num_episodes)):
            self.train_episode(max_env_steps, reward_network)

            if self.training_i % self.play_interval == 0:
                self.play(max_env_steps, reward_network=reward_network)

class QSoftActorCritic:
    """Implementation of soft actor critic."""

    def __init__(
            self,
            env,
            replay_buffer,
            feature_extractor,
            buffer_sample_size=10**4,
            gamma=0.99,
            learning_rate=3e-4,
            tbx_writer=None,
            entropy_tuning=False,
            entropy_target=1.0,
            tau=0.005,
            log_alpha=-2.995,
            q_net=None,
            play_interval=1,
    ):
        self.env = env
        self.feature_extractor = feature_extractor

        starting_state = self.env.reset()

        if self.feature_extractor is not None:
            starting_state = self.feature_extractor.extract_features(starting_state)
        state_size = starting_state.shape[0]

        # buffer
        self.replay_buffer = replay_buffer
        self.buffer_sample_size = buffer_sample_size

        # NNs
        if not q_net:
            self.q_net = QNetwork(state_size, env.action_space.n, 256)
        else:
            self.q_net = q_net

        self.avg_q_net = copy.deepcopy(self.q_net)

        # Dummy policy, it's just q_net in disguise!
        self.policy = self.q_net

        # initialize weights of moving avg Q net
        copy_params(self.q_net, self.avg_q_net)
        self.q_net.to(DEVICE)
        self.avg_q_net.to(DEVICE)

        # set hyperparameters
        self.log_alpha = torch.tensor(log_alpha).to(DEVICE)
        self.log_alpha = self.log_alpha.detach().requires_grad_(True)
        self.gamma = gamma
        self.entropy_target = entropy_target
        self.tau = tau

        # training meta
        self.training_i = 0
        self.play_i = 0
        self.entropy_tuning = entropy_tuning
        self.play_interval = play_interval

        # optimizers
        self.learning_rate = learning_rate
        self.q_optim = Adam(self.q_net.parameters(), lr=learning_rate)
        self.alpha_optim = Adam([self.log_alpha], lr=1e-2)

        # tensorboardX settings
        if not tbx_writer:
            self.tbx_writer = SummaryWriter()
        else:
            self.tbx_writer = tbx_writer

    def select_action(self, state, alpha):
        """Generate an action based on state vector using current policy.

        :param state: Current state vector. must be Torch 32 bit float tensor.
        """
        softmax_over_actions = F.softmax(
            (1.0/alpha) * self.q_net(state),
            dim=-1
        )
        dist = Categorical(softmax_over_actions)
        action = dist.sample()

        return action, dist.log_prob(action), dist

    def populate_buffer(self, max_env_steps):
        """
        Fill in entire replay buffer with state action pairs using current
        policy.
        """
        while len(self.replay_buffer) < self.buffer_sample_size:
            self.play(max_env_steps)

    def reset_training(self):
        self.q_net.apply(reset_parameters)
        self.avg_q_net = copy.deepcopy(self.q_net)
        self.q_optim = Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.alpha_optim = Adam([self.log_alpha], lr=1e-2)

    def tbx_logger(self, log_dict, training_i):
        """Logs the tag-value pairs in log_dict using TensorboardX.

        :param log_dict: {tag:value} dictionary to log.
        :param training_i: Current training iteration.
        """
        for tag, value in log_dict.items():
            self.tbx_writer.add_scalar(tag, value, training_i)

    def train_episode(self, max_env_steps, reward_net=None):
        """Train Soft Actor Critic"""

        # Populate the buffer
        self.populate_buffer(max_env_steps)

        # weight updates
        replay_samples = self.replay_buffer.sample(self.buffer_sample_size)
        state_batch = torch.from_numpy(replay_samples[0]).to(DEVICE)
        action_batch = torch.from_numpy(replay_samples[1]).to(DEVICE)
        reward_batch = torch.from_numpy(replay_samples[2]).to(DEVICE)
        next_state_batch = torch.from_numpy(replay_samples[3]).to(DEVICE)
        dones = torch.from_numpy(replay_samples[4]).type(torch.long).to(DEVICE)

        # if reward net exists, replace reward batch with fresh one
        if reward_net:
            fresh_rewards = reward_net(next_state_batch).flatten()
            assert fresh_rewards.shape == reward_batch.shape
            reward_batch = fresh_rewards

        # alpha must be clamped with a minumum of zero, so use exponential.
        alpha = self.log_alpha.exp().detach()
        self.policy.alpha = alpha

        with torch.no_grad():
            # Figure out value function
            next_actions, log_next_actions, _ = self.select_action(
                next_state_batch,
                alpha
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
        q_values = get_action_q(q_a_values, action_batch.squeeze())
        q_loss = F.mse_loss(q_values, q_target)

        # policy loss
        _, log_actions, action_dist = self.select_action(state_batch, alpha)

        # update parameters
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

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
                'loss/alpha loss': alpha_loss.item(),
                'Q/avg_q_target': q_target.mean().item(),
                'Q/avg_q': q_values.mean().item(),
                'Q/avg_reward': reward_batch.mean().item(),
                'Q/avg_V': next_state_values.mean().item(),
                'pi/avg_entropy': action_dist.entropy().mean(),
                'pi/avg_log_actions': log_actions.detach().mean().item(),
                'alpha': alpha.item(),
            },
            self.training_i
        )

        self.training_i += 1

    def play(self, max_env_steps, reward_network=None, render=False, best_action=False):
        """
        Play one complete episode in the environment's gridworld.
        Automatically appends to replay buffer, and logs with Tensorboardx.

        :param max_env_steps: Maximum number of steps to take in playthrough.
        :param reward_network: Replaces environment's builtin rewards.
        :param render: If True, renders the playthrough.
        :param best_action: If True, uses best actions instead of stochastic actions.
        """

        done = False
        total_reward = np.zeros(1)
        state = self.env.reset()
        if self.feature_extractor is not None:
            state = self.feature_extractor.extract_features(state)
        episode_length = 0

        for _ in range(max_env_steps):
            # Env returns numpy state so convert to torch
            torch_state = torch.from_numpy(state).type(torch.float32)
            torch_state = torch_state.to(DEVICE)

            alpha = self.log_alpha.exp().detach()

            # select an action to do
            if best_action:
                action = self.policy.eval_action(torch_state)
            else:
                action, _, _ = self.select_action(torch_state, alpha)

            next_state, reward, done, _ = self.env.step(action.item())
            next_state = self.feature_extractor.extract_features(next_state)

            if render:
                self.env.render()

            if reward_network:
                reward = reward_network(torch_state).cpu().item()

            if episode_length > max_env_steps:
                self.replay_buffer.push((
                    state,
                    action.cpu().numpy(),
                    reward,
                    next_state,
                    done
                ))
            else:
                self.replay_buffer.push((
                    state,
                    action.cpu().numpy(),
                    reward,
                    next_state,
                    not done
                ))

            state = next_state
            total_reward += reward
            episode_length += 1

            if done:
                break

        self.tbx_writer.add_scalar(
            'rewards/episode_reward',
            total_reward.item(),
            self.play_i
        )

        self.tbx_writer.add_scalar(
            'rewards/episode_length',
            episode_length,
            self.play_i
        )

        self.play_i += 1

    def train(self, num_episodes, max_env_steps, reward_network=None):
        """Train and play environment every play_interval, appending obtained
        states, actions, rewards, and dones to the replay buffer.

        :param num_episodes: number of episodes to train for.
        :param play_interval: trainig episodes between each play session.
        """

        print("Training RL . . .")

        for _ in tqdm(range(num_episodes)):
            self.train_episode(max_env_steps, reward_network)

            if self.training_i % self.play_interval == 0:
                self.play(max_env_steps, reward_network=reward_network)

"""
Soft actor critic per "Soft Actor-Critic Algorithms and Applications" Haarnoja
et. al 2019.
"""
import sys
import copy
import operator
import functools
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np

from tensorboardX import SummaryWriter

sys.path.insert(0, "..")
from neural_nets.base_network import BaseNN  # NOQA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def is_degenerate(ten):
    """Returns true if ten contains inf or nan.

    :param ten: input tensor to check for degeneracies.
    """
    contains_nan = torch.isnan(ten).any()
    contains_inf = (ten == float("inf")).any()

    return contains_nan or contains_inf


def soften_distribution(probs, alpha):
    """Apply additive smoothing to discrete probabilities.

    :param probs: Probabilities to smooth.
    :param alpha: additive factor.
    """
    soft = probs + alpha
    soft = soft / soft.sum(dim=1, keepdim=True)

    return soft


class QNetwork(BaseNN):
    """Q function network."""

    def __init__(self, state_length, action_space, hidden_layer_width):
        super().__init__()

        action_length = functools.reduce(operator.mul, action_space.shape)

        #q1
        self.q1_in_layer = nn.Linear(
            state_length + action_length, hidden_layer_width
        )
        self.q1_hidden1 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.q1_hidden2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.q1_head = nn.Linear(hidden_layer_width, 1)

        #q2
        self.q2_in_layer = nn.Linear(
            state_length + action_length, hidden_layer_width
        )
        self.q2_hidden1 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.q2_hidden2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.q2_head = nn.Linear(hidden_layer_width, 1)
        self.to(DEVICE)

    def forward(self, states, actions):
        state_action = torch.cat([states, actions], dim=1)

        #q1
        q1 = F.relu(self.q1_in_layer(state_action))
        q1 = F.relu(self.q1_hidden1(q1))
        q1 = F.relu(self.q1_hidden2(q1))
        q1 = self.q1_head(q1)

        #q2
        q2 = F.relu(self.q2_in_layer(state_action))
        q2 = F.relu(self.q2_hidden1(q2))
        q2 = F.relu(self.q2_hidden2(q2))
        q2 = self.q2_head(q2)

        return q1, q2


class PolicyNetwork(BaseNN):
    """Policy network for soft actor critic."""

    def __init__(self, state_size, action_space, hidden_layer_width):
        super().__init__()

        # properties inferred from action space
        self.action_scale = (
            torch.tensor((action_space.high - action_space.low) / 2.0)
            .to(DEVICE)
            .to(torch.float)
        )

        self.action_bias = (
            torch.tensor((action_space.high + action_space.low) / 2.0)
            .to(DEVICE)
            .to(torch.float)
        )

        action_width = functools.reduce(operator.mul, action_space.shape)

        self.in_layer = nn.Linear(state_size, hidden_layer_width)
        self.hidden1 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.hidden2 = nn.Linear(hidden_layer_width, hidden_layer_width)

        # params for Gaussian
        self.means = nn.Linear(hidden_layer_width, action_width)
        self.log_stds = nn.Linear(hidden_layer_width, action_width)

        self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        means = self.means(x)
        log_stds = self.log_stds(x)
        log_stds = torch.clamp(log_stds, min=-20.0, max=2.0)

        return means, log_stds

    def sample(self, state):
        """Generate an action based on state vector using current policy.

        :param state: Current state vector. must be Torch 32 bit float tensor.
        """
        dist = self.action_distribution(state)
        raw_action = dist.rsample()  # reparametrization trick

        # enforcing action bounds
        tanh_action = torch.tanh(raw_action)  # prevent recomputation later.
        action = tanh_action * self.action_scale + self.action_bias

        # change of variables for log prob
        raw_log_prob = dist.log_prob(raw_action)
        log_prob = raw_log_prob - torch.log(
            self.action_scale * (1 - tanh_action.pow(2)) + FEPS
        )
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, dist

    def select_action(self, state):
        """Select action for playing. Selects mean action only.

        :param state: State to select action from.
        :return: produced action.
        """
        means, _ = self.__call__(state)
        action = self.action_scale * means + self.action_bias

        return action

    def action_distribution(self, state):
        """Returns a pytorch distribution object based on policy output.

        :param state: Input state vector.
        """
        means, stds = self.__call__(state)
        dist = Normal(means, torch.exp(stds))

        return dist


class SoftActorCritic:
    """Implementation of soft actor critic."""

    def __init__(
        self,
        env,
        replay_buffer,
        max_episode_length,
        feature_extractor,
        buffer_sample_size=10 ** 4,
        gamma=0.99,
        learning_rate=3e-4,
        tbx_writer=None,
        entropy_tuning=False,
        entropy_target=-2.0,
        tau=0.005,
        log_alpha=-2.995,
        policy_net=None,
        q_net=None,
        render=False,
    ):
        # env related settings
        self.env = env
        self.render = render
        self.feature_extractor = feature_extractor
        self.max_episode_length = max_episode_length
        starting_feature = self.env_reset()
        feature_size = starting_feature.shape[0]

        # buffer
        self.replay_buffer = replay_buffer
        self.buffer_sample_size = buffer_sample_size

        # NNs
        if not policy_net:
            self.policy = PolicyNetwork(feature_size, env.action_space, 256)
        else:
            self.policy = policy_net

        if not q_net:
            self.q_net = QNetwork(feature_size, env.action_space, 256)
        else:
            self.q_net = q_net

        self.avg_q_net = copy.deepcopy(self.q_net)

        # initialize weights of moving avg Q net
        copy_params(self.q_net, self.avg_q_net)

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

        # optimizers
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)
        self.q_optim = Adam(self.q_net.parameters(), lr=learning_rate)
        self.alpha_optim = Adam([self.log_alpha], lr=1e-2)

        # tensorboardX settings
        if not tbx_writer:
            self.tbx_writer = SummaryWriter("runs/generic_soft_ac")
        else:
            self.tbx_writer = tbx_writer

    def env_reset(self):
        """Reset environment, but return features from feature_extractor.

        :return: state(features)
        """
        state = self.env.reset()
        return self.feature_extractor.extract_features(state)

    def env_step(self, action):
        """Takes action in environment but returns features rather than
        states, from feature_extractor.

        :param action: action to take.
        :return: state(features), reward, done, info
        """
        state, reward, done, info = self.env.step(action)
        state = self.feature_extractor.extract_features(state)

        return state, reward, done, info

    def populate_buffer(self):
        """
        Fill in entire replay buffer with state action pairs using current
        policy.
        """
        while len(self.replay_buffer) < self.buffer_sample_size:
            self.play(self.max_episode_length)

    def tbx_logger(self, log_dict, training_i):
        """Logs the tag-value pairs in log_dict using TensorboardX.

        :param log_dict: {tag:value} dictionary to log.
        :param training_i: Current training iteration.
        """
        for tag, value in log_dict.items():
            self.tbx_writer.add_scalar(tag, value, training_i)

    def print_grads(self):
        for name, param in self.policy.named_parameters():
            print(name, param.grad.abs().sum())

    def train_episode(self):
        """Train Soft Actor Critic"""

        # Populate the buffer
        self.populate_buffer()

        # weight updates
        replay_samples = self.replay_buffer.sample(self.buffer_sample_size)
        state_batch = torch.from_numpy(replay_samples[0]).to(DEVICE)
        action_batch = torch.from_numpy(replay_samples[1]).to(DEVICE)
        reward_batch = torch.from_numpy(replay_samples[2]).to(DEVICE).unsqueeze(1)
        next_state_batch = torch.from_numpy(replay_samples[3]).to(DEVICE)
        dones = torch.from_numpy(replay_samples[4]).type(torch.long).to(DEVICE).unsqueeze(1)

        # alpha must be clamped with a minumum of zero, so use exponential.
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            # Figure out value function
            next_actions, log_next_actions, _ = self.policy.sample(next_state_batch)
            target_q1, target_q2 = self.avg_q_net(next_state_batch, next_actions)
            target_q = torch.min(target_q1, target_q2)
            next_state_values = target_q - alpha * log_next_actions

            # Calculate Q network target
            done_floats = dones.type(torch.float)
            q_target = reward_batch.clone()
            q_target += self.gamma * done_floats * next_state_values

        # Q net outputs values for all actions, so we index specific actions
        q1, q2 = self.q_net(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)

        # policy loss
        actions_pi, log_probs_pi, action_dist = self.policy.sample(
            state_batch
        )
        q1_pi, q2_pi = self.q_net(state_batch, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = ((alpha * log_probs_pi) - q_pi).mean()

        # update parameters
        self.q_optim.zero_grad()
        q1_loss.backward()
        self.q_optim.step()

        self.q_optim.zero_grad()
        q2_loss.backward()
        self.q_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # automatic entropy tuning
        alpha_loss = (
            self.log_alpha * (log_probs_pi + self.entropy_target).detach()
        )
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
                "loss/q1 loss": q1_loss.item(),
                "loss/q2 loss": q2_loss.item(),
                "loss/pi loss": policy_loss.item(),
                "loss/alpha loss": alpha_loss.item(),
                "Q/avg_q_target": q_target.mean().item(),
                "Q/avg_q1": q1.mean().item(),
                "Q/avg_q2": q2.mean().item(),
                "Q/avg_reward": reward_batch.mean().item(),
                "Q/avg_V": next_state_values.mean().item(),
                "H/alpha": alpha.item(),
                "H/pi_entropy": action_dist.entropy().mean(),
                "H/pi_log_pi": log_probs_pi.mean(),
            },
            self.training_i,
        )

        self.training_i += 1

    def play(self, max_steps, render=False):
        """
        Play one complete episode in the environment's gridworld.
        Automatically appends to replay buffer, and logs with Tensorboardx.
        """

        done = False
        total_reward = np.zeros(1)
        state = self.env_reset()
        episode_length = 0
        max_steps_elapsed = False

        while not (done or max_steps_elapsed):
            # Env returns numpy state so convert to torch
            torch_state = torch.from_numpy(state).type(torch.float32)
            torch_state = torch_state.to(DEVICE).unsqueeze(0)

            action, _, _ = self.policy.sample(torch_state)
            action = action.detach().cpu().numpy()
            action = action.reshape(self.env.action_space.shape)

            next_state, reward, done, _ = self.env_step(action)

            episode_length += 1

            max_steps_elapsed = episode_length > max_steps

            if max_steps_elapsed:
                self.replay_buffer.push(
                    (state, action, reward, next_state, done)
                )
            else:
                self.replay_buffer.push(
                    (state, action, reward, next_state, not done)
                )

            if render:
                self.env.render()

            state = next_state
            total_reward += reward

        self.tbx_writer.add_scalar(
            "rewards/episode_reward", total_reward.item(), self.play_i
        )

        self.tbx_writer.add_scalar(
            "rewards/episode_length", episode_length, self.play_i
        )

        self.play_i += 1

    def train_and_play(self, num_episodes, play_interval, halt_at_end=False):
        """Train and play environment every play_interval, appending obtained
        states, actions, rewards, and dones to the replay buffer.

        :param num_episodes: number of episodes to train for.
        :param play_interval: trainig episodes between each play session.
        """

        for _ in range(num_episodes):
            self.train_episode()

            if self.training_i % play_interval == 0:
                self.play(self.max_episode_length, self.render)

        if halt_at_end:
            import pdb; pdb.set_trace()

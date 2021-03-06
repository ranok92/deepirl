""" Define GCL family IRL methods. """
import warnings
from collections import namedtuple
import functools
import operator
import numpy as np

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from neural_nets.base_network import BaseNN, BasePolicy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple(
    "Transition",
    [
        "state",
        "action",
        "next_state",
        "reward",
        "done",
        "traj_end",
        "action_log_prob",
    ],
)
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)


def play(policy, env, max_steps, reward_net=None):
    """
    Plays using policy on environment for a maximum number of episodes.

    :param policy: Policy to use to play.
    :param env: Environment to play in.
    :param max_episodes: Maximum number of steps to take.
    :return: Buffer of standard transition named tuples.

    """

    buffer = []
    done = False
    ep_length = 1

    state = env.reset()

    while ep_length <= max_steps:
        torch_state = torch.from_numpy(state).to(torch.float).to(DEVICE)
        torch_state = torch_state.unsqueeze(dim=0)

        action, log_prob = policy.action_log_probs(torch_state)
        action = action.detach().cpu().numpy()
        action = action.reshape(env.action_space.shape)

        torch_action = torch.from_numpy(action).to(torch.float).to(DEVICE)
        torch_action = torch_action.unsqueeze(dim=0)

        log_prob = log_prob.detach().cpu().numpy()

        next_state, reward, done, _ = env.step(action)

        if reward_net:
            try:
                reward = reward_net(torch_state, torch_action)
            except TypeError:
                reward = reward_net(torch_state)

        max_steps_elapsed = ep_length > max_steps

        buffer.append(
            Transition(
                state,
                action,
                next_state,
                reward,
                not done if max_steps_elapsed else done,
                max_steps_elapsed,
                log_prob,
            )
        )

        state = next_state

        if done:
            break

        ep_length += 1

    return buffer


def bulk_torch_convert(tensors, torch_type):
    """
    Bulk convert list of tensors to desired torch type.

    :param tensors: Iterable of tensors.
    :param torch_type: torch type to convert tensors to.
    :return: tuple of converted tensors, in the order iterated over.
    :rtype: tuple of torch tensors.
    """
    out = []

    for tensor in tensors:
        out.append(tensor.to(torch_type))

    return tuple(out)


class BaseExpert:
    """Base class for expert trajectory generation/retrieval."""

    def get_expert_states(self):
        """Returns an numpy array of expert states."""
        raise NotImplementedError

    def get_expert_actions(self):
        """Returns an numpy array of expert actions."""
        raise NotImplementedError


class PolicyExpert(BaseExpert):
    """Generates expert trajectores from an expert policy."""

    def __init__(self, policy, env, num_trajs, max_episode_length):
        if not isinstance(policy, BasePolicy):
            warnings.warn("Given policy is not a BasePolicy instance.")

        assert policy is not None, "Policy is none!"
        self.policy = policy
        self.env = env

        self.expert_trajs = self.generate_expert_trajectories(
            num_trajs, max_episode_length
        )

    def generate_expert_trajectories(self, num_trajs, max_episode_length=1000):
        """Generate a state buffer of expert trajectories.

        :param num_trajs: Number of trajectories to generate.
        :type num_trajs: int
        :param max_episode_length: max length of episodes, defaults to 10000
        :type max_episode_length: int, optional
        :return: list of (s, a, s', r, done) tuples
        :rtype: list of tuples.
        """
        buffer = []
        for _ in range(num_trajs):
            traj_buffer = play(self.policy, self.env, max_episode_length)
            buffer.extend(traj_buffer)

        return buffer

    def get_expert_states(self):
        states = [transition.state for transition in self.expert_trajs]
        return np.array(states)

    def get_expert_actions(self):
        actions = [transition.action for transition in self.expert_trajs]
        return np.array(actions)


class RewardNetwork(BaseNN):
    """Simple 2 layer reward network."""

    def __init__(self, state_length, action_length, hidden_width):
        super().__init__()

        # define layers of NN
        self.linear1 = nn.Linear(state_length + action_length, hidden_width)
        self.linear2 = nn.Linear(hidden_width, hidden_width)
        self.head = nn.Linear(hidden_width, 1)

        self.to(DEVICE)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.head(x)

        return x


class NaiveGCL:
    """Implements 'Guided Cost Learning' but without using policy optimizer
    used by Finn. et. al"""

    def __init__(
        self,
        rl_method,
        env,
        expert_states,
        expert_actions,
        reward_net=None,
        learning_rate=1e-3,
        tbx_writer=None,
    ):
        """Initialize a Naive version of GCL.

        :param rl_method: a class that does RL.
        :type rl_method: Extension of BaseRL class.
        :param env: Gym-like environment.
        :type env: anyting that supports reset(), step() like gym.
        :param expert: expert that provides expert trajectories.
        :type expert: Extension of BaseExpert.
        :param reward_net: a reward network to train., defaults to None
        :type reward_net: Extension of BaseNN, optional
        """
        # RL related
        self.rl = rl_method
        self.env = env
        initial_state = env.reset()
        state_length = initial_state.shape[0]
        action_length = functools.reduce(
            operator.mul, self.env.action_space.shape
        )

        # IRL related
        assert expert_states.shape[0] == expert_actions.shape[0]

        self.expert_states = torch.from_numpy(expert_states)
        self.expert_states = self.expert_states.to(torch.float).to(DEVICE)

        self.expert_actions = torch.from_numpy(expert_actions)
        self.expert_actions = self.expert_actions.to(torch.float).to(DEVICE)

        # NNs
        if not reward_net:
            self.reward_net = RewardNetwork(state_length, action_length, 256)
        else:
            self.reward_net = reward_net

        self.reward_optim = Adam(
            self.reward_net.parameters(), lr=learning_rate, weight_decay=1e-3
        )

        # tensorboard related
        if tbx_writer:
            self.tbx_writer = tbx_writer
        else:
            tbx_comment = "irl_naive_gcl_only"
            self.tbx_writer = SummaryWriter(comment=tbx_comment)

        # learning meta
        self.irl_epoch = 0

    def train_reward_episode(self, num_sample_trajs, max_steps):
        """
        Trains the reward function for one episode.

        :param num_sample_trajs: Number of trajectories to sample from policy.
        :type num_sample_trajs: int

        """

        # calculate expert loss
        L_expert = self.reward_net(self.expert_states, self.expert_actions)
        L_expert = L_expert.mean()

        # compute policy generator loss
        L_pi = 0

        # samples trajectories from policy
        for _ in range(num_sample_trajs):
            samples = self.generate_traj(max_steps)
            trans_idx = {
                label: idx for idx, label in enumerate(samples[0]._fields)
            }
            assert samples, "no transitions sampled!"
            samples = list(map(list, zip(*samples)))

            pi_states = torch.tensor(samples[trans_idx["state"]]).to(DEVICE)
            pi_actions = torch.tensor(samples[trans_idx["action"]]).to(DEVICE)
            pi_rewards = torch.tensor(samples[trans_idx["reward"]]).to(DEVICE)
            pi_log_probs = torch.tensor(
                samples[trans_idx["action_log_prob"]]
            ).to(DEVICE).flatten()

            (
                pi_states,
                pi_actions,
                pi_rewards,
                pi_log_probs,
            ) = bulk_torch_convert(
                (pi_states, pi_actions, pi_rewards, pi_log_probs), torch.float
            )

            # compute importance sampling weight
            is_weight = torch.exp(pi_rewards.sum() - pi_log_probs.sum())
            is_weight = is_weight.detach()
            max_weight = torch.tensor([1e6]).to(DEVICE)
            is_weight = torch.min(max_weight, is_weight)
            rewards = self.reward_net(pi_states, pi_actions)
            L_pi += is_weight * rewards.sum()

            self.tbx_writer.add_scalar(
                "irl/is_weight",
                is_weight,
                num_sample_trajs * (self.irl_epoch - 1) + num_sample_trajs,
            )

        L_pi = L_pi.mean()

        # backprop total loss
        L_tot = L_expert - L_pi
        L_tot = -L_tot  # maximize objective

        self.reward_optim.zero_grad()
        L_tot.backward()
        self.reward_optim.step()

        # log errors
        self.tbx_writer.add_scalar("irl/Expert loss", L_expert, self.irl_epoch)
        self.tbx_writer.add_scalar(
            "irl/traj generator loss", L_pi, self.irl_epoch
        )
        self.tbx_writer.add_scalar("irl/total error", L_tot, self.irl_epoch)

    def train_policy_episode(self, num_episodes, max_episode_length):
        self.rl.train(num_episodes, max_episode_length, self.reward_net)

    def generate_traj(self, max_steps):
        """Generate a trajectory from policy.

        :param max_steps: Max allowed steps to take in environment.
        :type max_steps: int
        :return: list of transition tuples
        :rtype: list of tuples
        """
        return play(self.rl.policy, self.env, max_steps, self.reward_net)

    def train_episode(
        self, num_traj_per_episode, max_env_steps, num_policy_episodes
    ):
        # train reward
        self.train_reward_episode(num_traj_per_episode, max_env_steps)

        # train policy
        self.train_policy_episode(num_policy_episodes, max_env_steps)

        self.irl_epoch += 1

    def train(
        self,
        num_episodes,
        num_traj_per_episode,
        max_env_steps,
        policy_episodes_per_episode,
    ):
        """
        Train agent and reward.

        :param num_episodes: number of episodes to train for.
        :type num_episodes: int
        :param num_traj_per_episode: number of trajectories sampled from
        policy for training each episode.
        :param max_env_steps: Max number of environment steps the RL agent is
        allowed to take.
        :param policy_episodes_per_episode: Number of times policy is trained
        per times reward is trained.
        """
        for _ in range(num_episodes):
            self.train_episode(
                num_traj_per_episode,
                max_env_steps,
                policy_episodes_per_episode,
            )


class NaiveAIRL(NaiveGCL):
    def __init__(
        self,
        rl_method,
        env,
        expert_states,
        expert_actions,
        reward_net=None,
        learning_rate=1e-3,
        tbx_writer=None,
    ):

        super().__init__(
            rl_method,
            env,
            expert_states,
            expert_actions,
            reward_net=reward_net,
            learning_rate=learning_rate,
            tbx_writer=tbx_writer,
        )

    def train_reward_episode(self, num_sample_trajs, max_steps):
        """
        Trains the reward function for one episode.

        :param num_sample_trajs: Number of trajectories to sample from policy.
        :type num_sample_trajs: int

        """

        # calculate expert loss
        L_expert = self.reward_net(self.expert_states, self.expert_actions)
        L_expert = L_expert.mean()

        # compute policy generator loss
        L_pi = 0

        # samples trajectories from policy
        for traj_counter in range(num_sample_trajs):
            samples = self.generate_traj(max_steps)
            trans_idx = {
                label: idx for idx, label in enumerate(samples[0]._fields)
            }
            assert samples, "no transitions sampled!"
            samples = list(map(list, zip(*samples)))

            pi_states = torch.tensor(samples[trans_idx["state"]]).to(DEVICE)
            pi_actions = torch.tensor(samples[trans_idx["action"]]).to(DEVICE)
            pi_rewards = torch.tensor(samples[trans_idx["reward"]]).to(DEVICE)
            pi_log_probs = torch.tensor(
                samples[trans_idx["action_log_prob"]]
            ).to(DEVICE).flatten()

            (
                pi_states,
                pi_actions,
                pi_rewards,
                pi_log_probs,
            ) = bulk_torch_convert(
                (pi_states, pi_actions, pi_rewards, pi_log_probs), torch.float
            )

            # compute importance sampling weight
            is_weight = torch.exp(pi_rewards - pi_log_probs)
            is_weight = is_weight.detach()

            rewards = self.reward_net(pi_states, pi_actions)
            L_pi += (is_weight * rewards.flatten()).sum()

            # log every is_weight seperately
            self.tbx_writer.add_histogram(
                "irl/is_weight",
                is_weight,
                self.irl_epoch * num_sample_trajs + traj_counter,
            )

        L_pi /= num_sample_trajs

        # backprop total loss
        L_tot = L_expert - L_pi
        L_tot = -L_tot  # maximize objective

        self.reward_optim.zero_grad()
        L_tot.backward()
        self.reward_optim.step()

        # log errors
        self.tbx_writer.add_scalar("irl/Expert loss", L_expert, self.irl_epoch)
        self.tbx_writer.add_scalar(
            "irl/traj generator loss", L_pi, self.irl_epoch
        )
        self.tbx_writer.add_scalar("irl/total error", L_tot, self.irl_epoch)

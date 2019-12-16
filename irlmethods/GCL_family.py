""" Define GCL family IRL methods. """
import warnings
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from neural_nets.base_network import BaseNN, BasePolicy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple(
    "Transition",
    ["state", "action", "next_state", "done", "traj_end", "action_log_prob"],
)
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)


def play(policy, env, max_steps):
    """
    Plays using policy on environment for a maximum number of episodes.

    :param policy: Policy to use to play.
    :param env: Environment to play in.
    :param max_episodes: Maximum number of steps to take.
    :return: Buffer of standard transition named tuples.

    """

    buffer = []
    done = False
    state = env.reset()
    ep_length = 1

    while ep_length <= max_steps:

        state = torch.from_numpy(state)

        action, log_prob = policy.action_log_probs(state)
        action = action.detach().cpu().numpy()

        log_prob = log_prob.detach().cpu().numpy()

        next_state, reward, done, _ = env.step(action)

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

        if done:
            break

        ep_length += 1

    return buffer


class BaseExpert:
    """Base class for expert trajectory generation/retrieval."""

    def get_expert_trajectory(self):
        """Returns num_trajs number of trajectories as a list of tuples when
        called."""
        raise NotImplementedError


class PolicyExpert(BaseExpert):
    """Generates expert trajectores from an expert policy."""

    def __init__(self, policy, env, num_trajs, max_episode_length):
        if not isinstance(policy, BasePolicy):
            warnings.warn("Given policy is not a BasePolicy instance.")

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
        for _ in num_trajs:
            traj_buffer = play(self.policy, self.env, max_episode_length)
            buffer.extend(traj_buffer)

        return buffer

    def get_expert_trajectory(self):
        return self.expert_trajs


class RewardNetwork(BaseNN):
    """Simple 2 layer reward network."""

    def __init__(self, state_length, hidden_width):
        super().__init__()

        # define layers of NN
        self.linear1 = nn.Linear(state_length, hidden_width)
        self.linear2 = nn.Linear(hidden_width, hidden_width)
        self.head = nn.Linear(hidden_width, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.head(x))

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
        learning_rate=1e-4,
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

        # IRL related
        assert (
            expert_states.shape[0] == expert_actions.shape[0] + 1
        ), "Missing final state."
        extended_rewards = np.concat((expert_actions, np.zeros(1)), axis=0)
        extended_rewards = extended_rewards.astype(float)

        self.expert_sa = np.concat((expert_states, extended_rewards), axis=1)
        self.expert_sa = torch.from_numpy(self.expert_sa).device(DEVICE)

        # NNs
        if not reward_net:
            self.reward_net = RewardNetwork(state_length, 256)
        else:
            self.reward_net = reward_net

        self.reward_optim = Adam(self.reward_net.params(), lr=learning_rate)

    def train_reward_episode(self):
        # run expert trajectories
        expert_reward_means = self.reward_net(self.expert_sa).mean()

    def generate_trajs(self, num_trajs):
        pass

    def train(self):
        pass

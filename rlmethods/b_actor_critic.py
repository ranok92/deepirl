'''An environment independant actor critic method.'''
import argparse
import pdb
import os
import pathlib
import datetime
import copy

from itertools import count
from collections import namedtuple
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from gym_envs import np_frozenlake  # NOQA: E402
import utils  # NOQA: E402
from neural_nets.base_network import BaseNN

from rlmethods.termination import VarianceTermination



parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


class Policy(BaseNN):
    """Policy network"""

    def __init__(self, state_dims, action_dims, body_net = None):
        super(Policy, self).__init__()

        if body_net:
            self.graft(body_net)
        else:
            self.body = nn.Sequential(
                nn.Linear(state_dims, 128),
                nn.ReLU()
            )

        self.action_head = nn.Linear(128, action_dims)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.body(x)

        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def graft(self, body):
        """Grafts a deep copy of another neural network's body into this
        network. Requires optimizer to be reset after this operation is
        performed.

        :param body: body of the neural network you want grafted.
        """
        assert body is not None, 'NN body being grafted is None!'

        self.body = copy.deepcopy(body)


class ActorCritic:
    """Actor-Critic method of reinforcement learning."""

    def __init__(self, env, feat_extractor= None, policy=None, termination = None, gamma=0.99, render=False,
                 log_interval=100, max_episodes=0, max_ep_length=200,
                 reward_threshold_ratio=0.99):
        """__init__

        :param env: environment to act in. Uses the same interface as gym
        environments.
        """

        self.gamma = gamma
        self.render = render
        self.log_interval = log_interval
        self.max_episodes = max_episodes
        self.max_ep_length = max_ep_length
        self.reward_threshold_ratio = reward_threshold_ratio

        self.env = env
        self.feature_extractor = feat_extractor

        if termination is not None:
            self.termination=termination
        else:
            self.termination = VarianceTermination()


        if self.feature_extractor is None:

            state_size = env.reset().shape[0]

        else:

            state_size = self.feature_extractor.extract_features(env.reset()).shape[0]

        print("Actor Critic initialized with state size ",state_size)
        # initialize a policy if none is passed.
        if policy is None:
            self.policy = Policy(state_size, env.action_space.n)
        else:
            self.policy = policy

        # use gpu if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 'cpu')

        self.policy = self.policy.to(self.device)

        # optimizer setup
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.EPS = np.finfo(np.float32).eps.item()

    def select_action(self, state):
        """based on current policy, given the current state, select an action
        from the action space.

        :param state: Current state in environment.
        """

        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_actions.append(SavedAction(m.log_prob(action),
                                                     state_value))
        return action.item()

    def generate_trajectory(self, num_trajs, path):

        for traj_i in range(num_trajs):

            # action and states lists for current trajectory
            actions = []
            states = [self.env.reset()]

            done = False
            while not done:
                action = self.select_action(states[-1])
                actions.append(action)

                state, rewards, done, _ = self.env.step(action)
                states.append(state)

            actions_tensor = torch.tensor(actions)
            states_tensor = torch.stack(states)

            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            torch.save(actions_tensor,
                       os.path.join(path, 'traj%s.acts' % str(traj_i)))

            torch.save(states_tensor,
                       os.path.join(path, 'traj%s.states' % str(traj_i)))

    def finish_episode(self):
        """Takes care of calculating gradients, updating weights, and resetting
        required variables and histories used in the training cycle one an
        episode ends."""
        R = 0
        saved_actions = self.policy.saved_actions
        policy_losses = []
        value_losses = []
        rewards = []

        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)

        # if single rewards, do not normalize mean distribution
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.EPS)

        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)

            r_tensor = torch.tensor([r])

            if torch.cuda.is_available():
                r_tensor = r_tensor.cuda()

            value_losses.append(F.smooth_l1_loss(value, r_tensor))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()


        loss.backward()
        self.optimizer.step()

        #additional lines for loss based termination
        gradient_magnitude = [torch.norm(param.grad) for param in \
                self.policy.parameters()]
        gradient_magnitude = sum(gradient_magnitude)/len(gradient_magnitude)

        self.termination.add_loss(
                **{
                    "value_loss": torch.stack(value_losses).sum().item(),
                    "policy_loss": torch.stack(policy_losses).sum().item(),
                    "total_loss": loss.item()
                }
        )

        #FIXME Better more universal way to handle this? Also seperate episodes
        # from RL itself and move into termination
        self.termination.current_episode += 1

        del self.policy.rewards[:]
        del saved_actions[:]

    def train(self, rewardNetwork=None, featureExtractor=None, irl=False):
        """Train actor critic method on given gym environment."""

        running_reward = 0

        for i_episode in count(1):

            if self.feature_extractor is not None:

                state = self.feature_extractor.extract_features(
                            self.env.reset())
            else:
                state = self.env.reset()

            # number of timesteps taken
            t = 0

            # rewards obtained in this episode
            # ep_reward = self.max_ep_length
            ep_reward = 0

            for t in range(self.max_ep_length):  # Don't infinite loop while learning

                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)

                if self.feature_extractor is not None:
                    state = self.feature_extractor.extract_features(
                            state)

                if rewardNetwork is None:

                    reward = reward
                else:
                    reward = rewardNetwork(state)
                    reward = reward.item()

                ep_reward += reward

                if self.render:
                    self.env.render()

                self.policy.rewards.append(reward)

                if done:
                    break

            running_reward = running_reward * self.reward_threshold_ratio +\
                ep_reward * (1-self.reward_threshold_ratio)

            self.finish_episode()

            # if not in an IRL setting, solve environment according to specs
            if not irl:
                if i_episode % self.log_interval == 0:
                    print('Ep {}\tLast length: {:5d}\tAvg. reward: {:.2f}'.format(
                        i_episode, t, running_reward))
                    self.termination.print_debug()

                if self.termination.is_terminated():
                    break

                if running_reward >= 0.95:
                    plt.axvline(x=i_episode)

                # terminate if max episodes exceeded
                if i_episode > self.max_episodes and self.max_episodes > 0:
                    break
            else:
                assert self.max_episodes > 0

                if i_episode % self.log_interval == 0:
                    print('Ep {}\tLast length: {:5d}\tAvg. reward: {:.2f}'.format(
                        i_episode, t, running_reward))

                # terminate if max episodes exceeded
                if i_episode > self.max_episodes:
                    break

                #terminate based on loss
                if self.termination.is_terminated():
                    break

        return self.policy

    def train_episode(self, reward_acc, rewardNetwork=None, featureExtractor=None):
        """
        performs a single RL training iterations.
        """
        state = self.env.reset()

        # rewards obtained in this episode
        ep_reward = 0

        for t in range(self.max_ep_length):  # Don't infinite loop while learning
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)

            if rewardNetwork is None:
                reward = reward

            else:
                reward = rewardNetwork(state)
                reward = reward.item()

            ep_reward += reward

            g = self.gamma
            reward_acc.value = g * reward_acc.value + (1-g) * ep_reward

            self.policy.rewards.append(reward)

            if done:
                break

        self.finish_episode()

    def train_mp(
        self,
        n_jobs=1,
        reward_net=None,
        feature_extractor=None,
        irl=False,
        log_interval=100
    ):

        self.policy.share_memory()

        # share the reward network memory if it exists
        if reward_net:
            reward_net.share_memory()

        # TODO: The target method here is weirdly setup, where part of the
        # functionality MUST lie outside of any class. How to fix this?
        mp.spawn(
            train_spawnable,
            args=(self, reward_net, feature_extractor, irl),
            nprocs=n_jobs
        )

        return self.policy


def train_spawnable(process_index, rl, *args):
    print("%d process spawned." % process_index)
    rl.train(*args)


if __name__ == '__main__':
    args = parser.parse_args()

    _env = gym.make('FrozenLakeNP-v0')
    _env.seed(args.seed)
    torch.manual_seed(args.seed)

    model = ActorCritic(_env, gamma=args.gamma, render=args.render,
                        log_interval=args.log_interval)

    model.train()

'''An environment independant actor critic method.'''
import argparse
import pdb
from itertools import count
from collections import namedtuple
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import sys
sys.path.insert(0, '..')
from gym_envs import np_frozenlake  # NOQA: E402
import utils #NOQA: E402


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


class Policy(nn.Module):
    """Policy network"""

    def __init__(self, state_dims, action_dims):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(state_dims, 128)
        self.affine2 = nn.Linear(128, 128)
        self.affine3 = nn.Linear(128, 128)
        self.affine4 = nn.Linear(128, 128)

        self.action_head = nn.Linear(128, action_dims)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))

        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


class ActorCritic:
    """Actor-Critic method of reinforcement learning."""

    def __init__(self, env, policy=None, gamma=0.99, render=False,
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

        # decorate environment's functions to return torch tensors
        self.env.step = utils.step_torch_state()(self.env.step)
        self.env.reset = utils.reset_torch_state()(self.env.reset)

        # initialize a policy if none is passed.
        if policy is None:
            self.policy = Policy(env.reset().shape[0], env.action_space.n)
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

        # state = torch.from_numpy(state).float()
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_actions.append(SavedAction(m.log_prob(action),
                                                     state_value))
        return action.item()

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

            if torch.cuda.is_available:
                r_tensor = r_tensor.cuda()

            value_losses.append(F.smooth_l1_loss(value, r_tensor))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()

        loss.backward()
        self.optimizer.step()

        del self.policy.rewards[:]
        del self.policy.saved_actions[:]

    def train(self):
        """Train actor critic method on given gym environment."""

        # keeps running avg of rewards through episodes
        running_reward = 0

        # keeps histogram of states visited
        state_visitation_histogram = torch.zeros(self.env.reset().shape[0],
                                                dtype=torch.float32).cuda()

        for i_episode in count(1):
            state = self.env.reset()

            # if torch.cuda.is_available():
                # state = torch.from_numpy(state).cuda().type(dtype=torch.float32)

            state_visitation_histogram += state

            # number of timesteps taken
            t = 0

            # rewards obtained in this episode
            # ep_reward = self.max_ep_length
            ep_reward = 0

            for t in range(self.max_ep_length):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)

                state_visitation_histogram += state
                ep_reward += reward

                if self.render:
                    self.env.render()

                self.policy.rewards.append(reward)

                if done:
                    break

            running_reward = running_reward * self.reward_threshold_ratio +\
                    ep_reward * (1-self.reward_threshold_ratio)

            self.finish_episode()

            if i_episode % self.log_interval == 0:
                print('Ep {}\tLast length: {:5d}\tAvg. reward: {:.2f}'.format(
                    i_episode, t, running_reward))

            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time \
                      steps!".format(running_reward, t))
                break

            # terminate if max episodes exceeded
            if i_episode > self.max_episodes and self.max_episodes > 0:
                break


if __name__ == '__main__':
    args = parser.parse_args()

    _env = gym.make('FrozenLakeNP-v0')
    _env.seed(args.seed)
    torch.manual_seed(args.seed)

    model = ActorCritic(_env, gamma=args.gamma, render=args.render,
                        log_interval = args.log_interval)

    model.train()

'''An environment independant actor critic method.'''
import argparse
from itertools import count
from collections import namedtuple
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


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
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

class ActorCritic:
    """Actor-Critic method of reinforcement learning."""

    def __init__(self, env, policy=None, gamma=0.99):
        """__init__

        :param env: environment to act in. Uses the same interface as gym
        environments.
        """
        self.env = env
        self.gamma = gamma

        # initialize a policy if none is passed.
        if policy is None:
            self.policy = Policy()
        else:
            self.policy = policy

        # optimizer setup
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-2)
        self.EPS = np.finfo(np.float32).eps.item()

    def select_action(self, state):
        """based on current policy, given the current state, select an action
        from the action space.

        :param state: Current state in environment.
        """
        state = torch.from_numpy(state).float()
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
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.EPS)

        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + \
                torch.stack(value_losses).sum()

        loss.backward()
        self.optimizer.step()

        del self.policy.rewards[:]
        del self.policy.saved_actions[:]


    def train(self):
        """Train actor critic method on given gym environment."""

        running_reward = 10
        for i_episode in count(1):
            state = self.env.reset()

            # number of timesteps taken
            t = 0
            for t in range(10000):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                if args.render:
                    self.env.render()
                self.policy.rewards.append(reward)
                if done:
                    break

            running_reward = running_reward * 0.99 + t * 0.01
            self.finish_episode()

            if i_episode % args.log_interval == 0:
                print('Ep {}\tLast length: {:5d}\tAvg. length: {:.2f}'.format(
                    i_episode, t, running_reward))
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time \
                      steps!".format(running_reward, t))
                break


if __name__ == '__main__':
    args = parser.parse_args()

    _env = gym.make('CartPole-v0')
    _env.seed(args.seed)
    torch.manual_seed(args.seed)

    model = ActorCritic(_env, gamma=args.gamma)
    model.train()

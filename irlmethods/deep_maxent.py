
'''
Deep maxent as defined by Wulfmeier et. al.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import sys
import os
sys.path.insert(0, '..')
import utils  # NOQA: E402
from irlmethods.irlUtils import getStateVisitationFreq  # NOQA: E402


class RewardNet(nn.Module):
    """Reward network"""

    def __init__(self, state_dims):
        super(RewardNet, self).__init__()

        self.affine1 = nn.Linear(state_dims, 128)

        self.reward_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))

        return self.reward_head(x)

    def save(self, path):
        """Save the model.

        :param path: path in which to save the model.
        """
        model_i = 0

        # os.makedirs(path, parents=True, exist_ok=True)

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        while os.path.exists(os.path.join(path, '%s.pt' % model_i)):
            model_i += 1

        filename = os.path.join(path, '%s.pt' % model_i)

        torch.save(self.state_dict(), filename)

    def load(self, path):
        '''load the model.

        :param path: path from which to load the model.
        '''
        self.load_state_dict(torch.load(path))
        self.eval()


'''
***Passing the parameters for the RL block :
    Directly to the RL block in the experiment folder and not through the IRL block as before
'''


class DeepMaxent():
    def __init__(self, rlmethod=None, env=None, iterations=10, log_intervals=1):

        # pass the actual object of the class of RL method of your choice
        self.rl = rlmethod
        self.env = env

        self.max_episodes = iterations
        self.env.step = utils.step_torch_state()(self.env.step)
        self.env.reset = utils.reset_torch_state()(self.env.reset)

        self.reward = RewardNet(env.reset().shape[0])
        self.optimizer = optim.Adam(self.reward.parameters(), lr=3e-4)
        self.EPS = np.finfo(np.float32).eps.item()
        self.log_intervals = log_intervals

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def expert_svf(self):
        return irlUtils.expert_svf()

    def policy_svf(self):
        pass

    def calculate_grads(self, optimizer, stateRewards, freq_diff):

        optimizer.zero_grad()
        dotProd = torch.dot(stateRewards.squeeze(), freq_diff.squeeze())
        dotProd.backward()

    def train():
        '''
        Contains the code for the main training loop of the irl method. Includes calling the RL and
        environment from within

        '''
        expertdemo_svf = self.expert_svf()  # get the expert state visitation frequency

        for i in range(self.max_episodes):

            current_agent_policy = self.rlmethod.policy

            current_agent_svf = policy_svf(
                current_agent_policy, self.env.rows, self.env.cols, self.env.action_space)

            diff_freq = expertdemo_svf - current_agent_svf  # these are in numpy

            diff_freq = torch.from_numpy(diff_freq).to(
                self.device).type(self.dtype)

            # returns a tensor of size (no_of_states x 1)
            reward_per_state = getperStateReward(
                self.reward, self.env.rows, self.env.cols)

            calculate_grads(self.optimizer, reward_per_state, diff_freq)

            optimizer.step()

        pass

'''
Deep maxent as defined by Wulfmeier et. al.
'''
import pdb
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import sys
import os
sys.path.insert(0, '..')
import utils  # NOQA: E402
from irlmethods import irlUtils
# from irlmethods.irlUtils import getStateVisitationFreq  # NOQA: E402

from rlmethods.b_actor_critic import Policy


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
'''
    the parameters rlmethod and env take objects of rlmethod and env respectively

'''


class DeepMaxEnt():
    def __init__(self, traj_path, rlmethod=None, env=None, iterations=10, 
                log_intervals=1 , on_server = True):

        # pass the actual object of the class of RL method of your choice
        self.rl = rlmethod
        self.env = env

        self.max_episodes = iterations
        self.traj_path = traj_path

        # TODO: These functions are replaced in the rl method already, this
        # needs to be made independant somehow
        # self.env.step = utils.step_torch_state()(self.env.step)
        # self.env.reset = utils.reset_torch_state()(self.env.reset)

        self.reward = RewardNet(env.reset().shape[0])
        self.optimizer = optim.Adam(self.reward.parameters(), lr=1e-2)
        self.EPS = np.finfo(np.float32).eps.item()
        self.log_intervals = log_intervals

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.reward = self.reward.to(self.device)

        #making it run on server
        self.on_server = on_server

    def expert_svf(self):
        return irlUtils.expert_svf(self.traj_path).type(self.dtype)


    def policy_svf(self, policy, rows, cols, actions_space=5 , goalState= np.asarray([0,0])):
        return irlUtils.getStateVisitationFreq(policy, rows, cols, actions_space , goalState)

    def calculate_grads(self, optimizer, stateRewards, freq_diff):
        optimizer.zero_grad()
        dotProd = torch.dot(stateRewards.squeeze(), freq_diff.squeeze())
        dotProd.backward()

    def per_state_reward(self, reward_function, rows, cols):
        all_states = itertools.product(range(rows), range(cols))

        all_states = torch.tensor(list(all_states),
                                  dtype=torch.float).to(self.device)

        return reward_function(all_states)


    def plot(self, image):
        # display_reward = reward_per_state.detach().cpu().numpy()
        display_reward = image.detach().cpu().numpy()
        display_reward = display_reward.reshape(self.env.rows,
                                                self.env.cols)

        im = plt.imshow(display_reward)
        cb = plt.colorbar(im)
        plt.pause(1.0)
        cb.remove()

    def plotLoss(self,x_axis,lossList):

        plt.plot(x_axis, lossList)
        plt.draw()
        plt.pause(.0001)



    def train(self):
        '''
        Contains the code for the main training loop of the irl method. Includes calling the RL and
        environment from within

        '''

        #expertdemo_svf = self.expert_svf()  # get the expert state visitation frequency
        expert_policy = Policy(2,5)
        expert_policy.to(self.device)
        expert_policy.load('./saved-models/1.pt')
        expertdemo_svf = self.policy_svf( expert_policy,
                                                self.env.rows, self.env.cols, goalState = np.array([3,3]))
        lossList = []
        x_axis = []

        if not self.on_server:
            plt.figure(0)
        self.plot(torch.from_numpy(expertdemo_svf).type(self.dtype))

        for i in range(self.max_episodes):
            print('starting iteration %s ...'% str(i))

            # current_agent_policy = self.rl.policy
            current_agent_policy = self.rl.train(rewardNetwork=self.reward,
                                                irl=True)

           

            current_agent_svf = self.policy_svf( current_agent_policy,
                                                self.env.rows, self.env.cols, np.array([3,3]))

            if not self.on_server:
                plt.figure(3)
            self.plot(torch.from_numpy(current_agent_svf).type(self.dtype))
            diff_freq = torch.from_numpy(expertdemo_svf - current_agent_svf).type(self.dtype)

            diff_freq = diff_freq.to(self.device)

            # diff_freq = torch.from_numpy(diff_freq).to(
                # self.device).type(self.dtype)

            # returns a tensor of size (no_of_states x 1)
            reward_per_state = self.per_state_reward(
                self.reward, self.env.rows, self.env.cols)

            diffabs = diff_freq.abs().sum().item()
            print ('Loss :',diffabs)
            lossList.append(diffabs)
            x_axis.append(i)

            if not self.on_server:
                plt.figure(1)
          
            self.plotLoss(x_axis,lossList)

            plt.figure(2)
            self.plot(diff_freq)

            self.calculate_grads(self.optimizer, reward_per_state, diff_freq)

            self.optimizer.step()

            print('done')

        plt.show()

        return self.rewardNN

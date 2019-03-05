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

import pathlib

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
        x = F.elu(self.affine1(x))
   
        x = self.reward_head(x)

        return x

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
                log_intervals=1 , on_server = True, plot_save_folder=None):

        # pass the actual object of the class of RL method of your choice
        self.rl = rlmethod
        self.env = env

        self.max_episodes = iterations
        self.traj_path = traj_path

        self.plot_save_folder = plot_save_folder

        # TODO: These functions are replaced in the rl method already, this
        # needs to be made independant somehow
        # self.env.step = utils.step_torch_state()(self.env.step)
        # self.env.reset = utils.reset_torch_state()(self.env.reset)

        self.reward = RewardNet(env.reset().shape[0])
        self.optimizer = optim.Adam(self.reward.parameters(), lr=1e-1)
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


    def plot(self, images, titles, save_path=None):

        nrows = max(1,int(len(images)/2)+1)
        ncols = 2
        colorbars = []

        for image_idx, image in enumerate(images):
            plt.subplot(nrows, ncols, image_idx+1)
            plt.title(titles[image_idx])

            im = plt.imshow(image)
            colorbars.append(plt.colorbar(im))

        plt.pause(0.0001)

        # save the plot
        if save_path:
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

            img_i = 0

            while os.path.exists(os.path.join(save_path, '%s.png' % img_i)):
                img_i += 1

            filename = os.path.join(save_path, '%s.png' % img_i)

            plt.savefig(filename, bbox_inches='tight')

        for cb in colorbars:
            cb.remove()

    def plotLoss(self,x_axis,lossList):
        plt.plot(x_axis, lossList)
        plt.draw()
        plt.pause(.0001)

    def resetPolicy(self):

        newNN = Policy(2,5)
        newNN.to(self.device)
        self.rl.policy = newNN



    def train(self):
        '''
        Contains the code for the main training loop of the irl method. Includes calling the RL and
        environment from within

        '''

        #expertdemo_svf = self.expert_svf()  # get the expert state visitation frequency
        expert_policy = Policy(2,5)
        expert_policy.to(self.device)
        expert_policy.load('./saved-models/1.pt')
        expertdemo_svf = self.policy_svf( expert_policy, self.env.rows,
                                         self.env.cols,
                                         goalState = np.array([3,3]))
        lossList = []
        x_axis = []

        for i in range(self.max_episodes):
            print('starting iteration %s ...'% str(i))

            # current_agent_policy = self.rl.policy

            self.resetPolicy()

            current_agent_policy = self.rl.train(rewardNetwork=self.reward,
                                                irl=True)

            current_agent_svf = self.policy_svf( current_agent_policy,
                                                self.env.rows, self.env.cols,
                                                np.array([3,3]))

            diff_freq = torch.from_numpy(expertdemo_svf - current_agent_svf).type(self.dtype)
            diff_freq = diff_freq.to(self.device)

            # returns a tensor of size (no_of_states x 1)
            reward_per_state = self.per_state_reward(
                self.reward, self.env.rows, self.env.cols)

            diffabs = diff_freq.abs().sum().item()
            print ('Loss :',diffabs)
            lossList.append(diffabs)
            x_axis.append(i)

            # PLOT
            to_plot = []
            to_plot.append(diff_freq.cpu().numpy().reshape((10,10)))
            to_plot.append(expertdemo_svf.reshape((10,10)))
            to_plot.append(current_agent_svf.reshape((10,10)))
            to_plot.append(reward_per_state.cpu().detach().numpy().reshape((10,10)))

            to_plot_descriptions = []
            to_plot_descriptions.append('SVF difference (L)')
            to_plot_descriptions.append('expert SVF')
            to_plot_descriptions.append('policy SVF')
            to_plot_descriptions.append('Reward per state')

            self.plot(to_plot, to_plot_descriptions,
                      save_path=self.plot_save_folder)

            # GRAD AND BACKPROP
            self.calculate_grads(self.optimizer, reward_per_state, -diff_freq)

            self.optimizer.step()

            print('done')

        return self.reward

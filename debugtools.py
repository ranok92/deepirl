import itertools
import pdb
import os
import glob
import numpy as np
import sys
sys.path.insert(0, '..')


from irlmethods.deep_maxent import RewardNet

import math
from envs.gridworld import GridWorld
import torch
import pdb
from utils import to_oh
from irlmethods.irlUtils import toTorch

#for visual
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32



def getperStateReward(rewardNetwork, rows=10 , cols =10):

    stateRewardTable = np.zeros([rows,cols])
    '''
    the states are linearized in the following way row*cols+cols = col 
    of the state visitation freq table 
    '''
    for i in range(rows):
        for j in range(cols):
            state = np.asarray([i, j])
            state = to_oh(i*cols+j,rows*cols)
            stateRewardTable[i , j] = rewardNetwork(toTorch(state))


    return stateRewardTable




if __name__ == '__main__':

    r = 10
    c = 10
    env = GridWorld(display=False, obstacles=[np.asarray([1, 2])])
    reward_network = RewardNet(env.reset().shape[0])
    reward_network.load('./experiments/saved-models-rewards/0.pt')
    reward_network.eval()
    reward_network.to(DEVICE)
    
    reward_values = getperStateReward(reward_network, rows = 10 , cols = 10)
    plt.imshow(reward_values)
    plt.colorbar()
    plt.show()

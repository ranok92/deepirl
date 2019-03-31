import itertools
import pdb
import os
import glob
import numpy as np
import sys
sys.path.insert(0, '..')


from irlmethods.deep_maxent import RewardNet
from featureExtractor.gridworld_featureExtractor import LocalGlobal,SocialNav
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


def visualize_rewards_in_environment(env , reward_network , feature_extractor):

    rows = env.rows
    cols = env.cols

    board_reward = np.zeros((rows,cols))

    for r in range(rows):
        for c in range(cols):

            #accessing inner variables of the environment, untill I can come up
            #with something better 
            env.state['agent_state'] = np.asarray([r,c])
            
            state_feat = feature_extractor.extract_features(env.state)

            reward = reward_network(state_feat)

            board_reward[r,c] = reward


    return board_reward





if __name__ == '__main__':

    r = 10
    c = 10
    #initialize environment
    env = GridWorld(display=True, is_onehot = False, 
                    obstacles=[np.asarray([2,2]),np.asarray([7,4]),np.asarray([3,5]),
                                np.asarray([3,3]),np.asarray([3,7]),np.asarray([5,7])] , 
                    goal_state=np.asarray([5,5]))

    #initialize feature extractor
    feat = LocalGlobal(window_size = 3 , fieldList = ['agent_state','goal_state','obstacles'])
    #feat = SocialNav(fieldList = ['agent_state','goal_state'])
    
    #initialize reward network
    print(env.reset())
    reward_network = RewardNet(feat.extract_features(env.reset()).shape[0])
    reward_network.load('./experiments/saved-models-rewards/125.pt')
    reward_network.eval()
    reward_network.to(DEVICE)
    
    #run function
    reward_values = visualize_rewards_in_environment(env,reward_network, feat)
    plt.imshow(reward_values)
    plt.colorbar()
    plt.show()

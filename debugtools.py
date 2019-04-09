import itertools
import pdb
import os
import glob
import numpy as np
import sys
sys.path.insert(0, '..')

from rlmethods.b_actor_critic import ActorCritic
from irlmethods.deep_maxent import RewardNet
from featureExtractor.gridworld_featureExtractor import LocalGlobal,SocialNav,FrontBackSideSimple
import math
from envs.gridworld import GridWorld
import torch
import pdb
from utils import to_oh
from irlmethods.irlUtils import toTorch

import re
numbers = re.compile(r'(\d+)')


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


def visualize_rewards_in_environment(action,env , reward_network , feature_extractor):

    rows = env.rows
    cols = env.cols

    board_reward = np.zeros((rows,cols))

    c_start = 0
    c_step = 1

    r_start = 0
    r_step = 1

    if action=='up':
        c_start = cols-1
        c_step = -1

    if action=='down':
        pass

    if action=='left':

        c_start = cols-1
        c_step = -1

    if action=='right':

        pass

    r_end = rows-r_start-1
    c_end = cols-c_start-1


    for r in range(r_start,r_end,r_step):
        for c in range(c_start,c_end,c_step):

            #accessing inner variables of the environment, untill I can come up
            #with something better 

            if action=='down' or action=='up':
                env.state['agent_state'] = np.asarray([c,r])
            else:
                env.state['agent_state'] = np.asarray([r,c])
            
            state_feat = feature_extractor.extract_features(env.state)

            reward = reward_network(state_feat)

            if action=='down' or action=='up':
                board_reward[c,r] = reward
            else:
                board_reward[r,c] = reward


    return board_reward


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def plot_reward_across_policy_models(filename,feature_extractor = None,
                                    seed_list = [],
                                    iterations_per_model = 50):
    color_list = ['r','g','b','c','m','y','k']
    counter = 0
    for seed in seed_list:

        env = GridWorld(display=False, is_onehot= False,is_random =True,
                    rows =10,
                    cols =10,
                    seed = seed,
                    obstacles = [np.asarray([5,1]),np.array([5,9])],
                                
                    goal_state = np.asarray([1,5]))

        max_ep_length = 20

        rl_method = ActorCritic(env, feat_extractor= feature_extractor, gamma = 0.99,
                                max_ep_length=max_ep_length, log_interval=50)

        model_names = glob.glob(os.path.join(filename,'*.pt'))

        reward_across_models = []
        for policy_file in sorted(model_names,key=numericalSort):

            rl_method.policy.load(policy_file)

        
            reward_per_model = 0
            print('Loading file :',policy_file)

            for r in range(iterations_per_model):
                state = feature_extractor.extract_features(env.reset())
                reward_per_run = 0
                for t in range(max_ep_length):

                    action = rl_method.select_action(state)
                    state,reward,done,_ = env.step(action)
                    reward_per_run+=reward
                    state = feature_extractor.extract_features(state)

                reward_per_model+=reward_per_run


            reward_per_model/=iterations_per_model
            print('Average reward for the model:', reward_per_model)
            reward_across_models.append(reward_per_model)

        plt.plot(reward_across_models,color = color_list[counter])
        plt.draw()
        plt.pause(0.001)
        counter+=1
    plt.show()
    return reward_across_models


def generate_trajectories(policy_fname_list,feature_extractor = None):

    #list containing the points of trajectories of all the policies
    trajectory_point_master_list = []

    env = GridWorld(display=True, is_onehot= False,is_random =False,
            rows =10,
            cols =10,
            seed = 7,
            obstacles = [np.asarray([5,5])],
            goal_state = np.asarray([1,5]))

    max_ep_length = 20
    run_iterations = 20

    rl_method = ActorCritic(env, feat_extractor= feature_extractor, gamma = 0.99,
                                max_ep_length=max_ep_length, log_interval=50)

    for name in policy_fname_list:

        #ready the policy
        rl_method.policy.load(name)
        trajectory_point_policy = []

        heat_map = np.zeros((env.rows,env.cols))    

        for i in range(run_iterations):
            trajectory_point_run = []
            state = env.reset()
            heat_map[state['agent_state'][0],state['agent_state'][1]]+=1
            trajectory_point_run.append((state['agent_state'][0]*env.cellWidth,state['agent_state'][1]*env.cellWidth))
            state = feature_extractor.extract_features(state)
            for t in range(max_ep_length):

                action = rl_method.select_action(state)
                state,reward,done,_ = env.step(action)
                heat_map[state['agent_state'][0],state['agent_state'][1]]+=1
                trajectory_point_run.append((state['agent_state'][0]*env.cellWidth,state['agent_state'][1]*env.cellWidth))
                state = feature_extractor.extract_features(state)

            trajectory_point_policy.append(trajectory_point_run)


        trajectory_point_master_list.append(trajectory_point_policy)
        plt.figure()
        plt.imshow(heat_map)
        plt.clim(0,70)
        plt.colorbar()
        plt.draw()
        plt.pause(.001)
    plt.show()
    env.draw_trajectories(trajectory_point_master_list,rows,cols)


if __name__ == '__main__':

    r = 10
    c = 10
    #initialize environment

    #initialize feature extractor
    feat = LocalGlobal(window_size = 3 , fieldList = ['agent_state','goal_state','obstacles'])
    #feat = SocialNav(fieldList = ['agent_state','goal_state'])
    #feat = FrontBackSideSimple(thresh1 = 1,thresh2 = 2,
    #                            thresh3= 3, fieldList = ['agent_state','goal_state','obstacles'])
    #initialize reward network
    #print(env.reset())
    '''
    reward_network = RewardNet(feat.extract_features(env.reset()).shape[0])
    reward_network.load('./experiments/saved-models-rewards/50_fbs_simple.pt')
    reward_network.eval()
    reward_network.to(DEVICE)
    
    #run function
    actions = ['left','right','up','down']

    for act in actions:
        reward_values = visualize_rewards_in_environment(act,env,reward_network, feat)
        plt.figure(act)
        plt.imshow(reward_values)
        plt.colorbar()
        plt.show()
   

    plot_reward_across_policy_models("./experiments/saved-models/Run_info_localglobal_v3_1/",
                                seed_list = [1,2,3,4],
                                feature_extractor = feat,
                                iterations_per_model = 10)

    '''
    policy_name_list = ["./experiments/saved-models/loc_glob_v3.1_win_3.pt",
                        "./experiments/saved-models/loc_glob_v3-1_keep_left.pt"]

    generate_trajectories(policy_name_list,feature_extractor = feat)
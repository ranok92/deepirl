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

def plot_reward_across_policy_models(foldername,
                                    expert = None,
                                    feature_extractor = None,
                                    seed_list = [],
                                    iterations_per_model = 50,
                                    compare_expert = True):
    color_list = ['r','g','b','c','m','y','k']
    counter = 0

    reward_across_seeds = []
    xaxis = None
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

        model_names = glob.glob(os.path.join(foldername,'*.pt'))

        xaxis = np.arange(len(model_names))

        reward_exp = get_rewards_for_model(expert,env = env,
                                feature_extractor = feature_extractor,
                                rl_method = rl_method,
                                max_ep_length = max_ep_length,
                                iterations = iterations_per_model)




        reward_across_models = []
        reward_expert = []
        for policy_file in sorted(model_names,key=numericalSort):



            reward_per_model = get_rewards_for_model(policy_file,env = env,
                                feature_extractor = feature_extractor,
                                rl_method = rl_method,
                                max_ep_length = max_ep_length,
                                iterations = iterations_per_model)

            print('Average reward for the model:', reward_per_model)
            reward_across_models.append(reward_per_model)
            reward_expert.append(reward_exp)


        reward_across_seeds.append(reward_across_models)

    np_reward_across_seeds = np.array(reward_across_seeds)

    print(np_reward_across_seeds.shape)
    means_rewards = np.mean(np_reward_across_seeds, axis = 0)

    print ("the mean rewards :", means_rewards)

    print("The mean across all runs and seeds : ",np.mean(means_rewards))

    std_rewards = np.std(np_reward_across_seeds, axis = 0)

    print ('the std :', std_rewards)
    plt.xlabel('IRL iteration no.')
    plt.ylabel('Reward obtained')
    plt.plot(xaxis,means_rewards,color = color_list[counter],label='IRL trained agent')
    plt.fill_between(xaxis , means_rewards-std_rewards , 
                    means_rewards+std_rewards, alpha = 0.5, facecolor = color_list[counter])
    plt.plot(reward_expert, color = 'k' , label='Expert agent')
    plt.legend()
    plt.draw()
    plt.pause(0.001)
    plt.show()
    return reward_across_models

def get_rewards_for_model(policy_file,
                env= None,
                feature_extractor = None,
                rl_method = None,
                max_ep_length = 20,
                iterations = 50):

    rl_method.policy.load(policy_file)
    reward_per_model = 0
    print('Loading file :',policy_file)

    for r in range(iterations):
        state = feature_extractor.extract_features(env.reset())
        reward_per_run = 0
        for t in range(max_ep_length):

            action = rl_method.select_action(state)
            state,reward,done,_ = env.step(action)
            reward_per_run+=reward
            state = feature_extractor.extract_features(state)

        reward_per_model+=reward_per_run

    reward_per_model/=iterations

    return reward_per_model



    
def generate_trajectories(policy_fname_list,feature_extractor = None):

    #list containing the points of trajectories of all the policies
    trajectory_point_master_list = []

    env = GridWorld(display=False, is_onehot= False,is_random =False,
            rows =10,
            cols =10,
            seed = 7,
            obstacles = [np.asarray([5,5])],
            goal_state = np.asarray([1,5]))

    max_ep_length = 15
    run_iterations = 50

    rl_method = ActorCritic(env, feat_extractor= feature_extractor, gamma = 0.99,
                                max_ep_length=max_ep_length, log_interval=50)

    labels = ['0','1','2','3','4','5','6','7','8','9']

    for name in policy_fname_list:

        #ready the policy
        rl_method.policy.load(name)
        trajectory_point_policy = []

        env = GridWorld(display=False, is_onehot= False,is_random =False,
                        rows =10,
                        cols =10,
                        seed = 7,
                        obstacles = [np.asarray([5,5])],
                        goal_state = np.asarray([1,5]))

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
        fig,ax = plt.subplots()

        im = ax.imshow(heat_map,vmin=0,vmax=40)
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.set_xlabel('Columns of the gridworld',fontsize='large')
        ax.set_ylabel('Rows of the gridworld',fontsize='large')

        for i in range(len(labels)):
            for j in range(len(labels)):

                text = ax.text(j,i,heat_map[i,j], ha="center", va="center", color="black")

        #ax.set_title("Grid location visitation frequency for a unbiased agent")

        #plt.colorbar()
        #plt.clim(0,70)
        plt.draw()
        plt.pause(.001)
    plt.show()


if __name__ == '__main__':

    r = 10
    c = 10
    #initialize environment
    '''
    env = GridWorld(display=False, is_onehot= False,is_random =False,
                rows =10,
                cols =10,
                seed = 12,
                obstacles = [np.asarray([5,1]),np.array([5,9])],
                            
                goal_state = np.asarray([1,5]))
    '''
    #initialize feature extractor
    #feat = LocalGlobal(window_size = 3 , fieldList = ['agent_state','goal_state','obstacles'])
    #feat = SocialNav(fieldList = ['agent_state','goal_state'])
    feat = FrontBackSideSimple(thresh1 = 1,thresh2 = 2,
                                thresh3= 3, fieldList = ['agent_state','goal_state','obstacles'])
    #initialize reward network
    #print(env.reset())
    '''
    reward_network = RewardNet(feat.extract_features(env.reset()).shape[0])
    reward_network.load('./experiments/saved-models-rewards/30local_global.pt')
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
                                expert = './experiments/saved-models/loc_glob_v3.1_win_3.pt',
                                seed_list = [1,2,3,4,5],
                                feature_extractor = feat,
                                iterations_per_model = 30)



    '''
    policy_name_list = ["./experiments/saved-models/fbs_simple.pt",
                        "./experiments/saved-models/Run_info_fbs_keep_left/45.pt"]

    generate_trajectories(policy_name_list,feature_extractor = feat)
    
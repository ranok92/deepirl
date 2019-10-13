import itertools
import pdb
import os
import glob
import numpy as np
import sys
sys.path.insert(0, '..')


from utils import reset_wrapper, step_wrapper
from rlmethods.b_actor_critic import Policy

from rlmethods.b_actor_critic import ActorCritic
from irlmethods.deep_maxent import RewardNet
from featureExtractor.gridworld_featureExtractor import LocalGlobal,SocialNav,FrontBackSideSimple
import math
from envs.gridworld import GridWorld
from envs.gridworld_clockless import GridWorldClockless
from envs.gridworld_drone import GridWorldDrone
import torch
import pdb
from utils import to_oh
from irlmethods.irlUtils import toTorch

from irlmethods.irlUtils import expert_svf, get_svf_from_sampling
from irlmethods.irlUtils import get_states_and_freq_diff, calculate_expert_svf, calculate_svf_from_sampling
import re
numbers = re.compile(r'(\d+)')


#for visual
import matplotlib.pyplot as plt
import pickle

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


def visualize_rewards_from_reward_directory(directory_name, feature_extractor ,  env):

    #given a directory name, this function will read each of the reward network
    #from the directory and plots the reward for each of the actions and stores them
    #in different directories based on the action and directory name


    #create parent directory

    dir_name = directory_name.split('/')
    cdw = os.getcwd()

    create_dir_path = os.path.join(cdw,'experiments/plots/')

    parent_reward_directory = os.path.join(create_dir_path,dir_name[-1])
    try:  
        os.mkdir(parent_reward_directory)
    except OSError:  
        print ("Creation of the directory failed.")
    else:  
        print ("Successfully created the directory.")

    reward_network_names = glob.glob(os.path.join(directory_name,'*.pt'))

    actions = ['left','right','up','down']

    #create directories for reward plots obtained from each of the actions

    for act in actions:
            action_dir = act
            try:
                os.mkdir(os.path.join(parent_reward_directory,action_dir))
            except OSError:
                print("cant create directory")


    for network_fname in reward_network_names:

        network_number = network_fname.split('/')[-1].split('.')[0]

        reward_network = RewardNet(feat.extract_features(env.reset()).shape[0])
        reward_network.load(network_fname)
        reward_network.eval()
        reward_network.to(DEVICE)
        
        #run function

        for act in actions:

            dir_to_save = os.path.join(parent_reward_directory,act)
            fname = dir_to_save+'/'+network_number+'.png'
            reward_values = visualize_rewards_in_environment(act,env,reward_network, feat)
            plt.figure(act)
            plt.imshow(reward_values)
            plt.colorbar()
            plt.savefig(fname)
            plt.clf()
       


def visualize_rewards_in_environment(action,env , reward_network , feature_extractor):

    #given the action, the environment, the reward network and feature extractor this function
    #returns the reward for each grid location of the environment as depicted by the reward_network
    #The reward at a particular grid location is the reward the agents get by landing at that position 
    #using the action provided in 'action'.

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

    #given a folder of policy networks, the function will go through them one by one and
    #create a plot of the rewards obtained by each of the policy networks and compare them
    #to that of an expert (if provided)
    color_list = ['r','g','b','c','m','y','k']
    counter = 0

    reward_across_seeds = []
    xaxis = None
    for seed in seed_list:

        env = GridWorld(display=False, is_onehot= False,is_random =True,
                    rows =10,
                    cols =10,
                    seed = seed,

                    obstacles = [np.asarray([5,1]),np.array([5,9]),
                    np.asarray([4,1]),np.array([6,9]),
                    np.asarray([3,1]),np.array([7,9])],
                                
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

            print('asdfasfsa',policy_file)

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
    #given a policy file it returns the amount of rewards it will get across some runs.
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
    torch.cuda.empty_cache() 
    return reward_per_model



  
def generate_agent_grid_visitation_map(policy_fname_list,feature_extractor = None, store = False):

    #given the policy file name list and feature extractor creates a heatmap of the 
    #agent on the gridworld based on the trajectories in the list
    #if store=True, the figure is stored in the form of a pickle


    #list containing the points of trajectories of all the policies
    trajectory_point_master_list = []
    traj_to_plot = 2

    env = GridWorld(display=False, is_onehot= False,is_random =False,
            rows =10,
            cols =10,
            seed = 3,
            obstacles = [np.asarray([5,5])],
            goal_state = np.asarray([1,5]))

    max_ep_length = 15
    run_iterations = 50

    rl_method = ActorCritic(env, feat_extractor= feature_extractor, gamma = 0.99,
                                max_ep_length=max_ep_length, log_interval=50)

    labels = ['0','1','2','3','4','5','6','7','8','9']

    counter = 0
    for name in policy_fname_list:
        counter+=1
        if counter==traj_to_plot:
            policy_name_to_plot = name
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

        ax.set_xlabel('Columns of the gridworld', fontsize='large')
        ax.set_ylabel('Rows of the gridworld', fontsize='large')



        for i in range(len(labels)):
            for j in range(len(labels)):

                text = ax.text(j,i,heat_map[i,j], ha="center", va="bottom", color="black")
                #arrow = ax.arrow(j,i,.1,.1,shape='full',head_width= .2)
                #arrow = ax.annotate("",xy = (j,i) , arrowprops = arrow)
                pass
        ax.set_title("Grid location visitation frequency for a unbiased agent")

        #plt.colorbar()
        #plt.clim(0,70)
        plt.draw()
        if store:
            pickle_filename = 'FigureObject'+str(counter)+'.fig.pickle' 
            pickle.dump(fig,open(pickle_filename, 'wb'))
        plt.pause(.001)

    #annotate_trajectory(policy_name_to_plot, env, rl_method,
    #                    max_ep_length, ax, feature_extractor=feature_extractor)

    plt.show()



def compare_svf(expert_folder, agent_policy, env_reward=None, env=None, feat=None):
    '''
    expert folder - folder containing expert trajectories
    agent_policy_folder/policy - a folder or a single policy
    Given these two information, the compare_svf function 
    saves the svf for all the policies which can be used for visual comparison.
    env_reward is the reward network corresponding to the policy network
    '''

    dot_product_loss = []

    environment = env
    state_space = feat.extract_features(environment.reset()).shape[0]

    #plotting for the expert
    expert_svf_dict = calculate_expert_svf(expert_folder, 
                        feature_extractor=feat,
                        gamma=0.99)
    #pdb.set_trace()
    exp_arr = np.zeros(len(expert_svf_dict.keys()))
    i = 0
    exp_state_key = {}
    for key in expert_svf_dict.keys():
        exp_arr[i] = expert_svf_dict[key]
        exp_state_key[key] = i
        i+=1
    '''####################
    expert = np.squeeze(exp_arr)

    print('The expert shape', expert.shape)

    print('The sum :', np.sum(expert))
    plt.plot(expert)
   
    #plt.show()
    #expert_file_name = expert_folder.split('/')[-2]
    #plt.savefig('./experiments/svf_visual/'+expert_file_name+'.jpg')
    '''###############
    #plotting for the agents

    if os.path.isfile(agent_policy):
        
        policy = Policy(state_space, environment.action_space.n, hidden_dims=[256])
        policy.load(agent_policy)
        policy.eval()
        policy.to(DEVICE)

        agent_file_name = agent_policy.strip().split('/')[-1].split('.')[0]
        agent_svf_dict = calculate_svf_from_sampling(no_of_samples=500, env=environment,
                                          policy_nn=policy, reward_nn=None,
                                          episode_length=40, feature_extractor=feat,
                                          gamma=.99)
        agent_arr = np.zeros(len(expert_svf_dict.keys()))
        i = 0
        for key in agent_svf_dict.keys():
            if key in exp_state_key.keys():
                agent_arr[exp_state_key[key]] = agent_svf_dict[key]

        agent = np.squeeze(agent_arr)
        #print(np.linalg.norm(np.asarray(diff), 1))
        plt.plot(agent)
        plt.show()

        states, diff = get_states_and_freq_diff(expert_svf_dict, agent_svf_dict, feat)
        #pdb.set_trace()

        plt.plot(diff)
        plt.show()
        #plt.savefig('./experiments/svf_visual/'+agent_file_name+'.jpg')
        #plt.clf()
        print(np.linalg.norm(np.asarray(diff),1))
    if os.path.isdir(agent_policy):

        #read files from the directory
        model_names = glob.glob(os.path.join(agent_policy, '*.pt'))
        reward_names = glob.glob(os.path.join(env_reward, '*.pt'))
        reward_names = sorted(reward_names, key=numericalSort)
        counter = 0
        for name in sorted(model_names, key=numericalSort):

            #load the policy network
            policy = Policy(state_space, environment.action_space.n, hidden_dims=[256])
            print('Loading file:', name)
            policy.load(name)
            policy.eval()
            policy.to(DEVICE)

            #load the reward network
            reward_net_name = reward_names[counter]
            reward = RewardNet(state_space, hidden_dims=[256])
            print('Loading reward network :', reward_net_name)
            reward.load(reward_net_name)
            reward.eval()
            reward.to(DEVICE)
            counter+=1
            agent_file_name = name.split('/')[-1].split('.')[0]
            agent_svf_dict = calculate_svf_from_sampling(no_of_samples=3000, env=environment,
                                              policy_nn=policy, reward_nn=reward,
                                              episode_length=30, feature_extractor=feat,
                                              gamma=.99)
            states, diff = get_states_and_freq_diff(expert_svf_dict, agent_svf_dict, feat)

            #pdb.set_trace()

            plt.plot(diff)
            #plt.show()
            #diff_arr = np.zeros(len(expert_svf_dict.keys()))
            plt.savefig('./experiments/results/svf_visual/'+agent_file_name+'.jpg')

            '''
            agent_arr = np.zeros(len(expert_svf_dict.keys()))
            i = 0
            for key in agent_svf_dict.keys():
                if key in exp_state_key.keys():
                    agent_arr[exp_state_key[key]] = agent_svf_dict[key]

            agent = np.squeeze(agent_arr)
            plt.plot(expert, 'r')

            plt.plot(agent, 'b')
            #plt.show()
            plt.savefig('./experiments/svf_visual/'+agent_file_name+'.jpg')
            plt.clf()
            '''
            diff_arr = np.asarray(diff)
            svf_diff = np.linalg.norm(diff, 1)
            print('The SVF diff for this model:', svf_diff)
            dot_product_loss.append(svf_diff)
        
        plt.plot(dot_product_loss, 'g')
        plt.savefig('./experiments/svf_visual/dot_prod.jpg')
   

def get_trajectory_information(trajectory_folder, feature_extractor, plot_info=False):
    '''
    Information it provides: 
        1. A histogram of the direction in which the 
    goal is with respect to the agent across all the states in all the 
    trajectories.
        2. A histogram of the distances in which the obstacles were wrt 
    the agent in all the trajectories.
        3. A histogram of the orientation in which the obstacles were wrt 
    the agent in all the trajectories.
        4. A histogram on the closeness indicator saying how fast the agent 
    was moving towards the goal

    **THIS METHOD IS SPECIFICALLY DESIGNED TO CATER TO THE NEEDS 
    OF THE FEATURE EXTRACTOR FrontBackSideSimple (and FrontBackSide in future)
    '''

    #initialize the histograms
    goal_orientation_hist = np.zeros(9)
    obs_orientation_hist = np.zeros(4)
    obs_dist_hist = np.zeros(5)
    closeness_indicator_hist = np.zeros(3)

    xaxis_9 = np.arange(9)
    xaxis_4 = np.arange(4)
    xaxis_3 = np.arange(3)
    xaxis_5 = np.arange(5)

    #read trajectories from the folder
    actions = glob.glob(os.path.join(trajectory_folder, '*.acts'))
    states = glob.glob(os.path.join(trajectory_folder, '*.states'))

    counter = 0
    for idx, state_file in enumerate(states):

        torch_traj = torch.load(state_file, map_location=DEVICE)
        traj_np = torch_traj.cpu().numpy()

        for i in range(traj_np.shape[0]):
            goal_orientation_hist += traj_np[i][0:9]
            closeness_indicator_hist += traj_np[i][9:12]

            orientation_dist_arr = traj_np[i][12:-1].reshape([4,4])
            obs_dist_hist[0:4] += orientation_dist_arr.sum(axis=1)
            obs_dist_hist[4] += traj_np[i][-1] #adding the hit flag
            obs_orientation_hist += orientation_dist_arr.sum(axis=0)
            counter += 1

    #normalizing the histograms based on the number of steps and the 
    #number of trajectories available
    goal_orientation_hist /= counter
    obs_orientation_hist /= counter
    closeness_indicator_hist /= counter
    obs_dist_hist /= counter

    #plot information
    if plot_info:

        plt.figure(0)
        plt.title('Goal orientation information.')
        plt.bar(xaxis_9, goal_orientation_hist)
        plt.figure(1)
        plt.title('Closeness inidicator information')
        plt.bar(xaxis_3, closeness_indicator_hist)
        plt.figure(2)
        plt.title('Orientation information')
        plt.bar(xaxis_4, obs_orientation_hist)
        plt.figure(3)
        plt.title('Distance from obstacles information')
        plt.bar(xaxis_5, obs_dist_hist)

        plt.show()

    return goal_orientation_hist, closeness_indicator_hist,\
           obs_orientation_hist, obs_dist_hist



def compile_results(reward_info, unknown_state_info, subject_info=None):

    '''
    reward_info : a list containing rewards obtained by the runs
    unknown_state_info : a list of unknown states encountered in each run
    subject_info : a list of subjects. The subject which the agent replaced during that
                   run
    '''
    good_runs = []
    meh_runs = []
    crashed_runs = []
    
    sub_traj_dict = {}
    if subject_info is not None:
        counter = 0
        for subject in subject_info:
            if subject not in sub_traj_dict:
                sub_traj_dict[subject] = {'reward' : [], 'unknown_state' : [], 
                                         'avg_reward':0, 'avg_unknown_states':0}
            sub_traj_dict[subject]['reward'].append(reward_info[counter])

            if reward_info[counter] > 1:
                good_runs.append(subject)
            elif reward_info[counter] < 0:
                crashed_runs.append(subject)
            else:
                meh_runs.append(subject)

            sub_traj_dict[subject]['unknown_state'].append(unknown_state_info[counter])
            counter+=1

        for subject in sub_traj_dict:

            sub_traj_dict[subject]['avg_reward'] = sum(sub_traj_dict[subject]['reward'])/len(sub_traj_dict[subject]['reward'])
            sub_traj_dict[subject]['avg_unknown_states'] = sum(sub_traj_dict[subject]['unknown_state'])/len(sub_traj_dict[subject]['unknown_state'])
        
        print("##############################")
        print('Run information Subject-wise:')

        for sub in sub_traj_dict:
            print('Subject :{}'.format(sub))
            print('Run information :', sub_traj_dict[sub])
        print("##############################")
        print('Good runs:', good_runs)
        print('Bad runs:', crashed_runs)
        print('Meh runs:', meh_runs)
        print('Fraction of good runs : {:.3f}'.format(len(good_runs)/(len(reward_info))))
        print("##############################")

    print('Overall results :')
    avg_reward = sum(reward_info)/len(reward_info)
    avg_unknown_states = sum(unknown_state_info)/len(reward_info)
    print('Average reward obtained over {:d} runs : {:.3f}'.format(len(reward_info), avg_reward))
    print('Average unknown states encountered over {:d} runs : {:.3f}'.format(len(reward_info), avg_unknown_states))
    return avg_reward, len(good_runs)/len(reward_info)


if __name__ == '__main__':

    r = 10
    c = 10
    #initialize environment
    
    '''
    env = GridWorld(display=False, is_onehot= False,is_random =False,
                rows =10,
                cols =10,
                seed = 12,
                obstacles = [np.asarray([5,5])],
                            
                goal_state = np.asarray([1,5]))
    '''
    annotation_file = './envs/expert_datasets/university_students/annotation/processed/frame_skip_1/students003_processed.txt'

    step_size = 10
    agent_width = 10
    grid_size = 10
    obs_width = agent_width
    '''
    env = GridWorldDrone(display=False, is_onehot = False, 
                        seed=999, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        annotation_file=annotation_file,
                        subject=None,
                        tick_speed=90, 
                        obs_width=10,
                        step_size=step_size,
                        agent_width=agent_width,
                        train_exact=True,
                        show_comparison=True,                       
                        rows=576, cols=720, width=grid_size)
    '''

    env = GridWorldClockless(display=False, is_random = True,
                    rows = 100, cols = 100,
                    agent_width=agent_width,
                    step_size=step_size,
                    obs_width=obs_width,
                    width=grid_size,
                    obstacles= './envs/map3.jpg',
                    goal_state=None, 
                    step_wrapper=step_wrapper,
                    seed=6651,
                    consider_heading=False,
                    reset_wrapper=reset_wrapper,
                    is_onehot = False)
    

    #initialize feature extractor
    feat = LocalGlobal(window_size=3, grid_size=grid_size,
                       agent_width=agent_width, 
                       obs_width=obs_width,
                       step_size=step_size,
                       )
    #feat = SocialNav(fieldList = ['agent_state','goal_state'])
    #feat = FrontBackSideSimple(fieldList = ['agent_state','goal_state','obstacles'])
    expert_folder = './experiments/trajs/ac_loc_glob_rectified_win_3_static_map3/'
    agent_policy = './experiments/results/Testing_new_env_quadra-reg-0-seed-6651-lr-0.001/saved-models/'
    agent_reward = './experiments/results/Testing_new_env_quadra-reg-0-seed-6651-lr-0.001/saved-models-rewards/'

    compare_svf(expert_folder, agent_policy, env_reward=agent_reward, env=env, feat=feat)
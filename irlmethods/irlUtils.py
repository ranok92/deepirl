# contains the calculation of the state visitation frequency method for the
# gridworld environment
import itertools
import pdb
import os
import glob
import numpy as np
import sys
sys.path.insert(0, '..')
from neural_nets.base_network import BaseNN
from torch.distributions import Categorical
from featureExtractor.gridworld_featureExtractor import OneHot,LocalGlobal,SocialNav,FrontBackSideSimple
from utils import reset_wrapper, step_wrapper
from rlmethods.b_actor_critic import Policy
from rlmethods.b_actor_critic import ActorCritic
import math
from envs.gridworld import GridWorld
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils import to_oh

import collections
from scipy import signal
import re
#for visual
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
'''

All the methods here are created with environments with relatively small statespace.
It is also assumed that we know all the possible states.
'''

class RewardNet(BaseNN):
    """Reward network"""

    def __init__(self, state_dims):
        super(RewardNet, self).__init__()

        self.affine1 = nn.Linear(state_dims, 128)

        self.reward_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.elu(self.affine1(x))

        x = self.reward_head(x)

        return x

def createStateActionTable(policy , rows= 10 , cols=10 , num_actions = 4):
    '''
    given a particular policy and info about the environment on which it is trained
    returns a matrix of size A x S where A is the
    size of the action space and S is the size of the state space
    things are hard coded here for the gridworld method but you can fiddle
    with the size of the environment
    '''
    stateActionTable = np.zeros([num_actions, (rows*cols)])
    '''
    the states are linearized in the following way row*cols+cols = col
    of the state visitation freq table
    '''
    for i in range(rows):
        for j in range(cols):
            state = np.asarray([i, j]) 
            state = to_oh( i*cols+j , rows*cols)
            action, _ = policy(toTorch(state))
            stateActionTable[:, i*cols+j] = toNumpy(action)

    return stateActionTable


def createStateTransitionMatix(rows=10, cols=10, action_space=5):
    '''
    Creates a matrix of dimension (total_states X total_actionsX total_states)
    As the environment is deterministic all the entries here should be either 
    0 or 1
    TranstionMatirx[next_state , action , prev_state]
    '''
    total_states = rows*cols
    transitionMatrix = np.zeros([total_states,action_space,total_states])
    for i in range(total_states):

        #check for boundary cases before making changes
        #check left if true then not boundary case

        if math.floor(i/cols)==math.floor((i-1)/cols):
            transitionMatrix[i-1,3,i] = 1
        else:
            transitionMatrix[i,3,i] = 1
        #check right
        if (math.floor(i/cols)==math.floor((i+1)/cols)):
            transitionMatrix[i+1,1,i] = 1
        else:
            transitionMatrix[i,1,i] = 1
        #check top
        if (i-cols)/cols >= 0:
            transitionMatrix[i-cols,0,i] = 1
        else:
            transitionMatrix[i,0,i] = 1
        #check down
        if (i+cols)/cols < rows:
            transitionMatrix[i+cols,2,i] = 1
        else:
            transitionMatrix[i,2,i] = 1

        transitionMatrix[i,4,i] = 1

    return transitionMatrix


def toTorch(nparray):
    torchTensor = torch.from_numpy(nparray)
    torchTensor = torchTensor.type(dtype).to(DEVICE)
    return torchTensor


def toNumpy(torchTensor):
    return torchTensor.to("cpu").detach().numpy()



def smoothing_over_state_space(state, feat, window):
    '''
    given a state this function returns a distribution over nearby states using a smoothing function0

    the total number of obstacles before and after the smoothing remains the same
    
    '''
    state_list = []
    sum_weights = 0
    global_rep = state[0:feat.gl_size+feat.rl_size]
    #print(type(state))
    spatial_rep = feat.state_to_spatial_representation(state)
    total_obs = int(np.sum(spatial_rep))
    conv = signal.convolve2d(spatial_rep, window, 'same')
    conv_flatten = conv.flatten()
    non_zero_list = []
    if total_obs > 0:
        for i in range(conv_flatten.shape[0]):
            if conv_flatten[i] > 0:
                non_zero_list.append(i)


        for combo in (itertools.combinations(non_zero_list, total_obs)):

            temp_local = np.zeros(conv_flatten.shape)
            weight = 1
            for obs in combo:

                temp_local[obs] = 1
                weight *= conv_flatten[obs]

            sum_weights += weight
            comb_state = np.concatenate((global_rep, temp_local))
            #state_list.append([np.reshape(temp_local, (3, 3)), weight])
            state_list.append([comb_state, weight]) 

        for entry in state_list:
            entry[1] /= sum_weights
    else:

        state_list.append((state, 1))

    return state_list



def getStateVisitationFreq(policy, rows=10, cols=10, num_actions=5,
                            goal_state = np.asarray([3,3]), episode_length = 30):
    '''
    The state visitation frequency for a given policy is the probability of being
    at a particular state at a particular time for the agent given:

            1. its policy (which changes the stateActionTable)
            2. the state transition matrix (property of the environment)
            3. The start state distribution (property of the environment)

    State visitation frequency will be a matrix of size:
            total_states x time
    where at the last step summation over time will
    reduce the state visitation frequency to a matrix of size :
            total_states x 1
    '''

    TIMESTEPS = episode_length
    TOTALSTATES = rows*cols
    GOALSTATE = (goal_state[0]*cols)+goal_state[1]

    stateVisitationMatrix = np.zeros([TOTALSTATES, TIMESTEPS])
    env = GridWorld(display=False, 
    				obstacles=[np.asarray([3, 7])],
    				goal_state = goal_state)

    '''
    The lines below were necessary if a policy dictionary file was passed as an argument to the 
    function. But if an actual policy (neural network) is passed in the arguments, the lines below will
    not be necessary.

    policy = Policy(env.reset().shape[0], env.action_space.n)
    policy.load_state_dict(torch.load(policyfile, map_location=DEVICE))
    policy.eval()
    policy.to(DEVICE)

    '''
    stateActionTable = createStateActionTable(
        policy, rows, cols, env.action_space.n)
    stateTransitionMatrix = createStateTransitionMatix(
        rows=rows, cols=cols, action_space=env.action_space.n)

    # considering all the states equally likely
    startStateDist = np.ones(TOTALSTATES)
    startStateDist = startStateDist/(TOTALSTATES)

    # calculating the state visitation freq
    for i in range(TIMESTEPS):

        for s in range(TOTALSTATES):
            # start state
            #stateVisitationMatrix[GOALSTATE,i] = 0
            if i == 0:
                stateVisitationMatrix[s, i] = startStateDist[s]
            else:
                for s_prev in range(TOTALSTATES):
                    for a in range(env.action_space.n):

                        stateVisitationMatrix[s, i] += stateVisitationMatrix[s_prev, i-1] * \
                            stateTransitionMatrix[s, a, s_prev] * \
                            stateActionTable[a, s_prev]

    return np.sum(stateVisitationMatrix,axis=1)/TIMESTEPS


#This is a more general function and should work with any
#state representation provided the state dictionary
#corresponding to that state representation is provided

def expert_svf(traj_path, feat=None, gamma=0.99):

    actions = glob.glob(os.path.join(traj_path, '*.acts'))
    states = glob.glob(os.path.join(traj_path, '*.states'))
    state_dict = feat.state_dictionary


    # histogram to accumulate state visitations
    svf = np.zeros((1,len(state_dict.keys())))

    for idx, state_file in enumerate(states):

        # traj_svf stores the state hist
        traj_hist = np.zeros((1,len(state_dict.keys())))
        #load up a trajectory and convert it to numpy
        torch_traj = torch.load(state_file, map_location=DEVICE)
        traj_np = torch_traj.cpu().numpy()

        #iterating through each of the states 
        #in the trajectory
        for i in range(traj_np.shape[0]):

            #this is for onehot

            #convert state to state index
            state_str = np.array2string(traj_np[i])
            #print('The state: ', traj_np[i][12:])
            #print('The local representation :', feat.state_to_spatial_representation(traj_np[i]))
            #print (len(state_dict.keys()))
            state_index = state_dict[state_str]

            # +1 for that index in the trajectory histogram
            traj_hist[0,state_index]+=1*math.pow(gamma,i)


        # normalize each trajectory over timesteps
        traj_hist/=np.sum(traj_hist)

        # accumulate frequencies through trajetories

        svf += traj_hist

    #normalize the svf over trajectories
    svf /= len(states)

    return svf


def calculate_expert_svf(traj_path, feature_extractor=None, gamma=0.99):
    '''
    Does the state visitation frequency calculation without creating a dictionary or storing the 
    entire state space.

    returns a dictionary where the keys are only the states that the expert has seen
    and the coressponding value is its visitation.
    '''
    actions = glob.glob(os.path.join(traj_path, '*.acts'))
    states = glob.glob(os.path.join(traj_path, '*.states'))

    # histogram to accumulate state visitations
    svf = {}
    for idx, state_file in enumerate(states):


        #load up a trajectory and convert it to numpy
        torch_traj = torch.load(state_file, map_location=DEVICE)
        traj_np = torch_traj.cpu().numpy()

        #iterating through each of the states 
        #in the trajectory
        for i in range(traj_np.shape[0]):

            #this is for onehot
            state_hash = feature_extractor.hash_function(traj_np[i])
            if state_hash not in svf.keys():
                svf[state_hash] = 1*math.pow(gamma,i)
            else:
                svf[state_hash] += 1*math.pow(gamma,i)

    #normalize the svf
    total_visitation = 0
    for state in svf.keys():

        total_visitation += svf[state]

    for state in svf.keys():

        svf[state] /= total_visitation

    return collections.OrderedDict(sorted(svf.items()))


def calculate_expert_svf_with_smoothing(traj_path, 
                                        smoothing_window=None, 
                                        feature_extractor=None, 
                                        gamma=0.99):

    actions = glob.glob(os.path.join(traj_path, '*.acts'))
    states = glob.glob(os.path.join(traj_path, '*.states'))

    # histogram to accumulate state visitations
    svf = {}

    for idx, state_file in enumerate(states):


        #load up a trajectory and convert it to numpy
        torch_traj = torch.load(state_file, map_location=DEVICE)
        traj_np = torch_traj.cpu().numpy()

        #iterating through each of the states 
        #in the trajectory
        for i in range(traj_np.shape[0]):

            #print(traj_np[i])
            #print('The state :', traj_np[i][-9:].reshape([3,3]))
            smoothened_list = smoothing_over_state_space(traj_np[i], feature_extractor, smoothing_window)
            #print('The smooth-list :')
            #for j in smoothened_list:
                #print(j[0][-9:].reshape([3, 3]),j[1])

            #input('press 1')
            for val in smoothened_list:
                #val is a list of size 2: val[0] : the state vector val[1] = 
                state_hash = feature_extractor.hash_function(val[0])
                if state_hash not in svf.keys():
                    svf[state_hash] = val[1]*math.pow(gamma,i)
                else:
                    svf[state_hash] += val[1]*math.pow(gamma,i)

    #normalize the svf
    total_visitation = 0
    for state in svf.keys():

        total_visitation += svf[state]

    for state in svf.keys():

        svf[state] /= total_visitation

    return collections.OrderedDict(sorted(svf.items()))





#function for sanity check for the states generated in the trajectories in the traj_path 
# mainly for debugging/visualizing the states in the path.
def debug_custom_path(traj_path, criteria ,state_dict = None):

	actions = glob.glob(os.path.join(traj_path, '*.acts'))
	states = glob.glob(os.path.join(traj_path, '*.states'))


	if criteria=='distance':
		histogram_bin = np.zeros(3)
		xaxis = np.arange(3)
	if criteria=='orientation':
		histogram_bin = np.zeros(12)
		xaxis = np.arange(4)
	if criteria=='both':
		histogram_bin = np.zeros(12)
		xaxis = np.arange(12)
	for idx, state_file in enumerate(states):

	    # traj_svf stores the state hist
	    traj_hist = np.zeros((1,len(state_dict.keys())))
	    #load up a trajectory and convert it to numpy
	    torch_traj = torch.load(state_file, map_location=DEVICE)
	    traj_np = torch_traj.cpu().numpy()

	    #iterating through each of the states 
	    #in the trajectory


	    for i in range(traj_np.shape[0]):

	    	#this is for onehot
	    	if criteria=='distance':

	    		if np.sum(traj_np[i][12:16]) > 0:

	    			histogram_bin[0]+=1
	    		if np.sum(traj_np[i][16:20]) > 0:
	    			histogram_bin[1]+=1
	    		if np.sum(traj_np[i][20:24]) > 0:

	    			histogram_bin[2]+=1

	    	else:

	    		histogram_bin+= traj_np[i][12:24]

	if criteria=='orientation':

		orient_bin = np.zeros(4)
		for i in range(12):

			orient_bin[i%4]+=histogram_bin[i]
			histogram_bin = orient_bin


	print(histogram_bin)
	plt.bar(xaxis,histogram_bin)
	plt.show()

	



'''
The select_action is a duplicate the the select action in the b_actor_critic.py
file. But unlike that method, this does not store any of the actions or 
rewards.
***Open to suggestions for better way of implementing this***

'''

def select_action(policy,state):
    """based on current policy, given the current state, select an action
    from the action space.

    :param state: Current state in environment.
    """
    probs, state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    #return action.item()
    val, ind = torch.max(probs,0)
    return ind


#should calculate the SVF of a policy network by running the agent a number of times
#on the given environment. Once the trajectories are obtained, each of the trajectories 
#multiplied by a certain weight, w, where w = e^(r)/ sum(e^(r) for r of all the 
#trajectories played)

def get_svf_from_sampling(no_of_samples = 1000, env = None ,
                                                 policy_nn = None , reward_nn = None,
                                                 episode_length = 20, feature_extractor = None, gamma=.99):


    num_states = 100
    #initialize the variable to store the svfs
    if feature_extractor is None:
        svf_policy = np.zeros((num_states, no_of_samples)) #as the possible states are 100
    else:
        #kept for later
        num_states = len(feature_extractor.state_dictionary.keys())
        svf_policy = np.zeros((num_states, no_of_samples))


    rewards = np.zeros(no_of_samples)
    approx_Z = 0 #this should be the sum of all the rewards obtained
    eps = 0.000001 # so that the reward is never 0.

    #start_state contains the distribution of the start state
    #as of now the code focuses on starting from all possible states uniformly

    start_state = np.zeros(num_states)
    '''
    need a non decreasing function that is always positive.
    get the range of rewards obtained and normalize it?
      The state space changes with the feature_extractor being used. 

    **Feature_extractor should have a member dictionary containing
    all possible states.

    The default feature for the environment for now is onehot.

    '''

    xaxis = np.arange(num_states)
    for i in range(no_of_samples):

        run_reward = 0
        state = env.reset()

        if feature_extractor is not None:
                state = feature_extractor.extract_features(state)

        if 'torch' in state.type():
                state_np = state.cpu().numpy()
        else:
            state_np= state
        if feature_extractor is None:
            #onehot
            state_index = np.where(state_np==1)[0][0]
        else:
            #feature_extractor wraps the state in torch tensor so convert that back
            state_index = feature_extractor.state_dictionary[np.array2string(state_np)]

        start_state[state_index]+=1
        #********************** till here************

        #np.where(state_np==1)[0][0] returns the state index
        #from the state representation

        svf_policy[state_index,i] = 1 #marks the visitation for 
                                                                                           #the state for the run

        for t in range(episode_length):
            action = select_action(policy_nn,state)
            state, reward, done,_ = env.step(action)
            #feature_extractor wraps the state in torch tensor so convert that back


            #get the state index
            if feature_extractor is None:
                state_index = np.where(state_np==1)[0][0]
            else:
                state = feature_extractor.extract_features(state)
                if 'torch' in state.type():
                    state_np= state.cpu().numpy()
                else:
                    state_np = state

                state_index = feature_extractor.state_dictionary[np.array2string(state_np)]


            svf_policy[state_index,i] += 1*math.pow(gamma,t) #marks the visitation for 
                                                                                       #the state for the run


            if reward_nn is not None:
                reward  = reward_nn(state)

            run_reward+=reward

        rewards[i]=run_reward
    #normalize the rewards to get the weights
    #dont want to go exp, thus so much hassle
    #print('Rewards untouched :',rewards)
    #rewards = rewards - np.min(rewards)+eps
    rewards = np.exp(rewards)
    #print('Rewards :',rewards)
    total_reward = sum(rewards)
    weights = rewards/total_reward
    #print('Weights from state_dict :',weights)
    #plt.plot(weights)
    #plt.draw()
    #plt.pause(0.001)
    #normalize the visitation histograms so that for each run the 
    #sum of all the visited states becomes 1




    norm_factor = np.sum(svf_policy,axis=0)

    svf_policy = np.divide(svf_policy, norm_factor)

    svf_policy = np.matmul(svf_policy,weights)

    return svf_policy



def calculate_svf_from_sampling(no_of_samples=1000, env=None,
                                policy_nn=None, reward_nn=None,
                                episode_length=20, feature_extractor=None,
                                gamma=0.99, scale_svf=True):
    
    '''
    calculating the state visitation frequency from sampling. This function
    returns a dictionary, where the keys consists of only the states that has been
    visited and their corresponding values are the visitation frequency
    '''
    eps = 0.000001
    if feature_extractor is None:
        print('Featrue extractor missing. Exiting.')
        return None

    rewards = np.zeros(no_of_samples)
    norm_factor = np.zeros(no_of_samples)

    svf_dict_list = []
    for i in range(no_of_samples):
        run_reward = 0
        current_svf_dict = {}
        state = env.reset()
        #print('agent position:', state['agent_state'])
        state = feature_extractor.extract_features(state)
        current_svf_dict[feature_extractor.hash_function(state)] = 1

        for t in range(episode_length):

            action = select_action(policy_nn, state)
            action = action.item()
            #print(action)
            state, reward, done,_ = env.step(action)
            #feature_extractor wraps the state in torch tensor so convert that back

            
            #get the state index

            state = feature_extractor.extract_features(state)
            if feature_extractor.hash_function(state) not in current_svf_dict.keys():
                current_svf_dict[feature_extractor.hash_function(state)] = 1*math.pow(gamma,t)
            else:
                current_svf_dict[feature_extractor.hash_function(state)] += 1*math.pow(gamma,t) 
                                                  
            if reward_nn is not None:
                reward  = reward_nn(state)

            run_reward+=reward

        rewards[i] = run_reward
        svf_dict_list.append(current_svf_dict)

    #rewards = rewards - np.min(rewards)+eps
    #changing it to the more generic exp
    #print('rewards non exp', rewards)
    rewards = np.exp(rewards)
    #print('Rewards :',rewards)
    total_reward = sum(rewards)

    #putting a control on the reweighting as discussed.
    if scale_svf:
        weights = rewards/total_reward
    else:
        weights = np.ones(no_of_samples)
    #print('weights from svf_dict:',weights)
    #plt.plot(weights)
    #plt.draw()
    #plt.pause(0.001)
    #merge the different dictionaries to a master dictionary and adjust the visitation 
    #frequencies according to the weights calculated

    for i in range(len(svf_dict_list)):

        dictionary = svf_dict_list[i]
        for key in dictionary:

            norm_factor[i] += dictionary[key]

    master_dict = {}
    #print ('The norm factor :', norm_factor)
    for i in range(len(svf_dict_list)):
        dictionary = svf_dict_list[i]
        for key in dictionary:

            if key not in master_dict:
                master_dict[key] = dictionary[key]*weights[i]/norm_factor[i]
            else:
                master_dict[key] += dictionary[key]*weights[i]/norm_factor[i]

    return collections.OrderedDict(sorted(master_dict.items()))



def calculate_svf_from_sampling_using_smoothing(no_of_samples=1000, env=None,
                                                policy_nn=None, reward_nn=None,
                                                episode_length=20, feature_extractor=None,
                                                gamma=0.99, window=None):
    
    '''
    calculating the state visitation frequency from sampling. This function
    returns a dictionary, where the keys consists of only the states that has been
    visited and their corresponding values are the visitation frequency
    '''
    eps = 0.000001
    if feature_extractor is None:
        print('Featrue extractor missing. Exiting.')
        return None

    rewards = np.zeros(no_of_samples)
    norm_factor = np.zeros(no_of_samples)

    svf_dict_list = []
    for i in range(no_of_samples):
        run_reward = 0
        current_svf_dict = {}
        state = env.reset()
        #print('agent position:', state['agent_state'])
        state = feature_extractor.extract_features(state)

        state_np = state.cpu().numpy()
        state_list = smoothing_over_state_space(state_np, feature_extractor, window)

        for item in state_list:
            #item = [ state , weight ]
            current_svf_dict[feature_extractor.hash_function(item[0])] = item[1]

        for t in range(episode_length):

            action = select_action(policy_nn, state)
            state, reward, done,_ = env.step(action)
            #feature_extractor wraps the state in torch tensor so convert that back

            
            #get the state index

            state = feature_extractor.extract_features(state)
            state_np = state.cpu().numpy()
            state_list = smoothing_over_state_space(state_np, feature_extractor, window)
            for item in state_list:
                if feature_extractor.hash_function(item[0]) not in current_svf_dict.keys():
                    current_svf_dict[feature_extractor.hash_function(item[0])] = item[1]*math.pow(gamma,t)
                else:
                    current_svf_dict[feature_extractor.hash_function(item[0])] += item[1]*math.pow(gamma,t) 
                                                  
            if reward_nn is not None:
                reward  = reward_nn(state)

            run_reward+=reward

        rewards[i] = run_reward
        svf_dict_list.append(current_svf_dict)

    #rewards = rewards - np.min(rewards)+eps
    #changing it to the more generic exp
    #print('rewards non exp', rewards)
    rewards = np.exp(rewards)
    #print('Rewards :',rewards)
    total_reward = sum(rewards)
    weights = rewards/total_reward
    #print('weights from svf_dict:',weights)
    #plt.plot(weights)
    #plt.draw()
    #plt.pause(0.001)
    #merge the different dictionaries to a master dictionary and adjust the visitation 
    #frequencies according to the weights calculated

    for i in range(len(svf_dict_list)):

        dictionary = svf_dict_list[i]
        for key in dictionary:

            norm_factor[i] += dictionary[key]

    master_dict = {}
    #print ('The norm factor :', norm_factor)
    for i in range(len(svf_dict_list)):
        dictionary = svf_dict_list[i]
        for key in dictionary:

            if key not in master_dict:
                master_dict[key] = dictionary[key]*weights[i]/norm_factor[i]
            else:
                master_dict[key] += dictionary[key]*weights[i]/norm_factor[i]

    return collections.OrderedDict(sorted(master_dict.items()))




def get_states_and_freq_diff(expert_svf_dict, agent_svf_dict, feat):
    '''
    takes in the svf dictionaries for the expert and the agent
    and returns two ordered lists containing the states visited and the 
    difference in their visitation frequency.
    '''
    state_list = []
    diff_list = []
    expert_iterator = 0
    agent_iterator = 0
    
    expert_key_list = list(expert_svf_dict.keys())
    agent_key_list = list(agent_svf_dict.keys())

    while True:

        #pdb.set_trace()
        exp_state = expert_key_list[expert_iterator]
        agent_state = agent_key_list[agent_iterator]
        if exp_state == agent_state:

            state_list.append(feat.recover_state_from_hash_value(exp_state))
            diff_list.append(expert_svf_dict[exp_state]-agent_svf_dict[agent_state])

            expert_iterator += 1
            agent_iterator += 1

        elif exp_state > agent_state:

            state_list.append(feat.recover_state_from_hash_value(agent_state))
            diff_list.append(0 - agent_svf_dict[agent_state])

            agent_iterator += 1

        elif agent_state > exp_state:

            state_list.append(feat.recover_state_from_hash_value(exp_state))
            diff_list.append(expert_svf_dict[exp_state] - 0)

            expert_iterator += 1

        if expert_iterator >= len(expert_key_list) or agent_iterator >= len(agent_key_list):

            if expert_iterator >= len(expert_key_list):

                while agent_iterator < len(agent_key_list):
                    
                    agent_state = agent_key_list[agent_iterator]
                    state_list.append(feat.recover_state_from_hash_value(agent_state))
                    diff_list.append(0 - agent_svf_dict[agent_state])
                    agent_iterator += 1

                break

            if agent_iterator >= len(agent_key_list):

                while expert_iterator < len(expert_key_list):
                    
                    exp_state = expert_key_list[expert_iterator]
                    state_list.append(feat.recover_state_from_hash_value(exp_state))
                    diff_list.append(expert_svf_dict[exp_state] - 0)
                    expert_iterator += 1

                break

    return state_list , diff_list

        



if __name__ == '__main__':



    r = 10
    c = 10

    #feat = FrontBackSideSimple(thresh1 = 1, thresh2 = 2,
    # 						   thresh3 = 3,
    #						   fieldList = ['agent_state','goal_state'])
    #feat = OneHot(grid_rows=10,grid_cols=10)
    np.random.seed(0)

    step_size = 3
    agent_width = 10
    grid_size = 10
    obs_width = agent_width  
    expert_folder = '../envs/expert_datasets/university_students/annotation/traj_info/frame_skip_1/students003/LocalGlobal_win5'

    feat = LocalGlobal(window_size=5, grid_size=grid_size,
                   agent_width=agent_width, 
                   obs_width=obs_width,
                   step_size=step_size,
                   )

    env = GridWorld(display=False, reset_wrapper=reset_wrapper, is_random = False,
    				step_wrapper=step_wrapper,
    				obstacles=[np.array([2,3])],
    				goal_state=np.array([5,5]),
    				is_onehot=False)

    
    #calculating the SVF for policy

    #calculating the svf for experts
    
    exp_svf = calculate_expert_svf(expert_folder, feature_extractor=feat)

    pdb.set_trace()
    print('printing from the expert :',exp_svf)


        #index = feat.state_dictionary[np.array2string(feat.recover_state_from_hash_value(key))]
        #print(index,' ',exp_svf[key],' ',exp_svf_3[0][index] , ' - ',exp_svf[key]-exp_svf_3[0][index]  )

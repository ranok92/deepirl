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
    '''
    state_list = []
    sum_weights = 0
    global_rep = state[0:feat.gl_size+feat.rl_size]
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


#will remove this once onehot becomes a feature extractor class
#but till then this stays 
def expert_svf_onehot(traj_path, ncols=10, nrows=10):

    actions = glob.glob(os.path.join(traj_path, '*.acts'))
    states = glob.glob(os.path.join(traj_path, '*.states'))

    # histogram to accumulate state visitations
    svf = np.zeros((1,ncols*nrows))

    for idx, state_file in enumerate(states):

        # traj_svf stores the state hist
        traj_hist = np.zeros((1,ncols*nrows))

        #load up a trajectory and convert it to numpy
        torch_traj = torch.load(state_file, map_location=DEVICE)
        traj_np = torch_traj.numpy()

        #iterating through each of the states 
        #in the trajectory
        for i in range(traj_np.shape[0]):

        	#this is for onehot

        	#convert state to state index
        	state_index = np.where(traj_np[i]==1)[0][0]

        	# +1 for that index in the trajectory histogram
        	traj_hist[0,state_index]+=1


        # normalize each trajectory
        traj_hist/=np.sum(traj_hist)

        # accumulate frequencies through time

        svf += traj_hist

    svf /= len(states)

    return svf


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
    return action.item()

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
	eps = 0.00001 # so that the reward is never 0.

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

		#for debugging purpose, shows the histogram of the start states 
		'''
		if i%50==0:
			plt.bar(xaxis,start_state)
			plt.draw()
			plt.pause(.001)
		'''
		#to make sure the start state is uniform among all possible states

		#the continues to restart unless the starting points are somewhat uniformly
		#distributed

		#unfortunately this only works for onehot representations 
		# for more generic features this needs to go
		'''
		while True:

			state = env.reset()
			if feature_extractor is not None:
				state = feature_extractor.extract_features(state)

			if 'torch' in state.type():
				state_np = state.cpu().numpy()
			else:
				state_np = state
			#from the one_hot state representation get the state
			#to get the index of the state

			#keeping this to make it work with the default case of onehot
			#**will be removed in the future
			if feature_extractor is None:
				state_index = np.where(state_np==1)[0][0]
			else:
				#feature_extractor wraps the state in torch tensor so convert that back

				state_index = feature_extractor.state_dictionary[np.array2string(state_np)]
			if start_state[state_index] > (no_of_samples/num_states)+2:
				pass
			else:
				start_state[state_index]+=1
				break
		'''
		#*******basically the same block as above without the 
		#constrain of making the starting point uniform 
		#because this will not be possible for all feature extractors
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

			if done:
				pass
				

			if t >= episode_length:
				rewards[i]=run_reward
				break

	#normalize the rewards to get the weights
	#dont want to go exp, thus so much hassle
	#print('Rewards untouched :',rewards)
	rewards = rewards - np.min(rewards)+eps
	#print('Rewards :',rewards)
	total_reward = sum(rewards)
	weights = rewards/total_reward
	#print('The weights :',weights)
	#normalize the visitation histograms so that for each run the 
	#sum of all the visited states becomes 1




	norm_factor = np.sum(svf_policy,axis=0)

	svf_policy = np.divide(svf_policy, norm_factor)

	svf_policy = np.matmul(svf_policy,weights)

	return svf_policy




if __name__ == '__main__':



    r = 10
    c = 10

    #feat = FrontBackSideSimple(thresh1 = 1, thresh2 = 2,
    # 						   thresh3 = 3,
    #						   fieldList = ['agent_state','goal_state'])
    #feat = OneHot(grid_rows=10,grid_cols=10)
    feat = LocalGlobal(window_size=3, fieldList=['agent_state','goal_state','obstacles'])
    env = GridWorld(display=False, reset_wrapper=reset_wrapper,
    				step_wrapper=step_wrapper,
    				obstacles=[np.array([3,2]), np.array([3,4])],
    				goal_state=np.array([5,5]),
    				is_onehot=False)


    state = feat.extract_features(env.reset())
    state = state.cpu().numpy()
    print(state)
    window = np.array([[0,.1,0],[.1,.6,.1],[0,.1,0]])
    v = smoothing_over_state_space(state,feat,window)

    print(v)
    '''
    state_space = feat.extract_features(env.reset()).shape[0]
    policy = Policy(state_space, env.action_space.n)
    policy.load('../experiments/saved-models/socialNav_g5_5_no_obs.pt')
    policy.eval()
    policy.to(DEVICE)
'''

    #debug_custom_path('../experiments/trajs/ac_gridworld_user_avoid',
    # 					'distance',state_dict = feat.state_dictionary)

    '''
    reward = RewardNet(state_space)
    reward.load('../experiments/saved-models-rewards/319.pt')
    reward.eval()
    reward.to(DEVICE)
	'''

    '''
    exp_svf = expert_svf('../experiments/trajs/ac_gridworld_socialNav_no_obs/',
     			state_dict = feat.state_dictionary)
    exp_svf = np.squeeze(exp_svf)
    print(exp_svf)
	

    
    #for other feature extractors
    xaxis = np.arange(len(feat.state_dictionary.keys()))
    plt.figure("the expert")
    plt.bar(xaxis,exp_svf)
    #plt.imshow()


    
    #exp_svf = expert_svf_onehot('../experiments/trajs/ac_gridworld/')

    #print ("The expert svf :", exp_svf)
    
    
    statevisit = getStateVisitationFreq(policy , rows = r, cols = c,
                                     num_actions = 5 , 
                                     goal_state = np.asarray([5,5]),
                                     episode_length = 20)

	
    statevisit3 = get_svf_from_sampling(no_of_samples = 3000 , env = env ,
						 policy_nn = policy , reward_nn = None,
						 episode_length = 20, feature_extractor = feat)

    statevisit3 = np.squeeze(statevisit3)
    plt.figure("the expert_samples")
    plt.bar(xaxis,statevisit3)
    #plt.imshow()
    plt.show()


    
    statevisitMat = np.resize(statevisit,(r,c))
    #statevisitMat2 = np.resize(statevisit2,(r,c))
    statevisitMat3 = np.resize(statevisit3,(r,c))

    #print ('svf :',statevisitMat2)
    plt.clf()
    plt.figure(0)
    plt.imshow(statevisitMat)
    plt.colorbar()
    plt.show()
   
    plt.figure(2)
    plt.imshow(statevisitMat3)
    plt.colorbar()
    plt.show()
    fname = './plots/'+str(3)+'.png'
    #plt.savefig(fname)
    #plt.clf()
    
    #print(stateactiontable)
    #print(np.sum(stateactiontable,axis=0))
    #mat = createStateTransitionMatix(rows=5,cols=5)
	'''


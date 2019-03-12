# contains the calculation of the state visitation frequency method for the
# gridworld environment
import itertools
import pdb
import os
import glob
import numpy as np
import sys
sys.path.insert(0, '..')

from rlmethods.b_actor_critic import Policy
from rlmethods.b_actor_critic import ActorCritic
import math
from envs.gridworld import GridWorld
import torch
import pdb



#for visual
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
'''

All the methods here are created with environments with relatively small statespace.
It is also assumed that we know all the possible states.
'''

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
    env = GridWorld(display=False, obstacles=[np.asarray([1, 2])])

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
            stateVisitationMatrix[GOALSTATE,i] = 0
            if i == 0:
                stateVisitationMatrix[s, i] = startStateDist[s]

            else:
                for s_prev in range(TOTALSTATES):
                    for a in range(env.action_space.n):

                        stateVisitationMatrix[s, i] += stateVisitationMatrix[s_prev, i-1] * \
                            stateTransitionMatrix[s, a, s_prev] * \
                            stateActionTable[a, s_prev]

    return np.sum(stateVisitationMatrix,axis=1)/TIMESTEPS


def expert_svf(traj_path, ncols=10, nrows=10):

    actions = glob.glob(os.path.join(traj_path, '*.acts'))
    states = glob.glob(os.path.join(traj_path, '*.states'))

    # histogram to accumulate state visitations
    state_hist = torch.zeros((1,ncols,nrows))

    svf = torch.zeros((ncols, nrows))

    for idx, state_file in enumerate(states):
        # load states from trajectory file
        torch_state = torch.load(state_file, map_location=DEVICE)

        # states can only be indecies if they are of type long
        state = torch_state.type(torch.long)

        # list of all time-slices of the histogram
        hist_slices = []

        for row_i, row in enumerate(state):
            # if this timestep has already been encountered, then increment in
            # existing histogram timeslice
            if row_i < len(hist_slices):
                hist_slices[row_i][row[0], row[1]] += 1

            # else initialize new histogram timeslice and append to hist_slices
            else:
                hist_slice = torch.zeros(ncols, nrows)
                hist_slice[row[0], row[1]] += 1
                hist_slices.append(hist_slice)

        # normalize each timestep
        for hist in hist_slices:
            hist /= hist.sum()

        # accumulate frequencies through time

        svf += torch.stack(hist_slices).sum(0)

    svf /= len(states)

    return svf.reshape(-1)

def getperStateReward(rewardNetwork, rows=10 , cols =10):

    stateRewardTable = np.zeros([(rows*cols),1])
    '''
    the states are linearized in the following way row*cols+cols = col 
    of the state visitation freq table 
    '''
    for i in range(rows):
        for j in range(cols):
            state = np.asarray([i, j])
            stateRewardTable[i*cols+j,1] = rewardNetwork(toTorch(state))


    return stateRewardTable



if __name__ == '__main__':

    r = 10
    c = 10
    env = GridWorld(display=False, obstacles=[np.asarray([1, 2])])
    print(env.reset())
    print(len(env.reset()))
    policy = Policy(env.reset().shape[0], env.action_space.n)
    policy.load_state_dict(torch.load('../experiments/saved-models/5.pt', map_location=DEVICE))
    policy.eval()
    policy.to(DEVICE)
    statevisit = getStateVisitationFreq(policy , rows = r, cols = c,
                                         num_actions = 5 , 
                                         goal_state = np.asarray([3,3]),
                                         episode_length = 20)
    print(type(statevisit))
    print(statevisit)
    statevisitMat = np.resize(statevisit,(r,c))
    statevisitMat/=5
    plt.imshow(statevisitMat)
    plt.colorbar()
    plt.show()
    #print(stateactiontable)
    #print(np.sum(stateactiontable,axis=0))
    #mat = createStateTransitionMatix(rows=5,cols=5)

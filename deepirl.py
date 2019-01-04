import numpy as np


'''
1. read expert demonstration - done 
2. calculate expert state visitation freq - done
3. calculate agent state visitation freq -  done
4. the irl block : takes in the state visitation freq of both the agent and the expert and spits out a new reward function?? 

'''

import ballenv_pygame as BE
import torch.nn.functional as F
from collections import OrderedDict
import torch
import torch.optim as optim


import numpy as np
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from collections import namedtuple
from rlmethods import ActorCritic
from networks import CostNetwork
from networks import Policy


#GLOBAL PARAMETERS

NN_PARAMS = {'input': 100,
             'hidden': [256 , 256],
             'output': 4    }

AGENT_RAD = 10

WINDOW_SIZE = 5
gamma = .99
log_interval = 1000
render = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

########################################################
#*****************utils to run the game taken from ballgameActorCritic.py******************




#******************************************************************************************


#*********************utils to run the IRL module************************

#demo file contains a list of states for multiple trajectories?
#trajectories - number of expert trajectories to be taken.
def takeExpertInfo(trajectories):

    env = BE.createBoard(display=True)
    actionList = [(0,-1),(1,0),(0,1),(-1,0)]
    mainStateInfo = []

    for i in range(trajectories):

        state = env.reset()
        state = env.get_state_BallEnv(window_size=5)
        done = False
        env.render()
        trajInfo = []
        trajInfo.append(state)
        while not done:

            action = env.take_action_from_userKeyboard()
            action = actionList[action]

            next_state , reward , done , _ = env.step(action)
            next_state = env.get_state_BallEnv()
            trajInfo.append(next_state)
            env.render()

        mainStateInfo.append(trajInfo)

    #need to divide the visitation frequency by the number of trajectories
    #used to get the visitation frequency
    np.save('expertstateinfolong_50.npy', mainStateInfo)
    return mainStateInfo


def getstateDict(str):
    #returns 2 things: a lookuptable : a dictionary where key is a string and corresponding value is a numpy array
                                        #describing the state
    #                  a statedict : a dictionary where the key is a string and value is an integer

    stateDict = OrderedDict()
    lookuptable = OrderedDict()
    if str=='no obstacle':



        for i in range(4):
            tempState = np.zeros(29)
            tempState[i] = 1           #corresponding to goal
            tempState[16] = 1          #corresponding to self

            lookuptable[np.array2string(tempState)] = tempState
            stateDict[np.array2string(tempState)] = i

    else:

        print 'Functionality has not been implemented yet.'

    return stateDict,lookuptable


#should return 2 things: a histogram over states that shows the frequency of each of the states
#and a reference that says which state is denoted by which of the indexes
def getStateVisitationFrequencyExpert(demofile,stateDict):

    info = np.load(demofile)
    total_trajs = len(info)
    maxlen = -1
    avglen = 0
    #find the max length of the given trajectories
    for traj in info:
        avglen+= len(traj)
        if maxlen < len(traj):
            maxlen = len(traj)
    print 'Averge length of trajectories :', float(avglen)/total_trajs
    #stateVisitationDict : a dictionary where keys are numpy arrays describing the states convereted to strings
    #and corresponding to each of the key is a numpy array of size = max length of the expert trajectories
    #storing the number of times that state is visited for that particular time
    stateVisitationDict = OrderedDict()

    for key in stateDict.keys():
        #initialize a numpy array of size (maxlen,) corresponding to each of the states
        stateVisitationDict[key] = 0
    #stateDict = OrderedDict()
    for trajinfo in info:

        for i in range(len(trajinfo)): #iterating through a single trajectory ,  i is basically the time
            temp = np.array2string(trajinfo[i])

            #storing the freq of the state
            if temp not in stateVisitationDict.keys():
                #initialize a numpy array of size (maxlen,) corresponding to each of the states
                stateVisitationDict[temp] = 0
            #storing which string represents which state
            if temp not in stateDict.keys():

                stateDict[temp] = trajinfo[i]
            stateVisitationDict[temp]+=1

    print 'state dict : ',stateDict.keys()
    print 'the state visitationdict :',stateVisitationDict
    print 'done'
    #state dict : a dictionary where key is a string and corresponding value is a numpy array
    #stateVisitationDict : a dictionary where key is a string and corresponding value is a int denoting the frequency

    #an array where the index is the state and the value in that index is the freq of that state
    #a dictionary where the key is an integer and the corresponding value is a numpy array that says which index in the
    #array denotes which state

    #we can use the fact that the order of keys entered will be same for both the dictionaries


    counter = 0
    freqarray = np.zeros([len(stateDict.keys()),1])
    stateLookup = OrderedDict()
    for i in range(len(stateDict.keys())):

        #print stateDict[i]
        freqarray[i] = stateVisitationDict[stateDict.keys()[i]]
        stateLookup[stateDict.keys()[i]] = i

    return np.divide(freqarray,total_trajs)


def getStateTensor(lookuptable):

    #the states are of shape (29,)
    stateShape = lookuptable[lookuptable.keys()[0]].shape
    no_of_states = len(lookuptable.keys())

    stateTensor = torch.zeros(no_of_states , stateShape[0])
    counter = 0
    for key in lookuptable.keys():

        tempState = lookuptable[key]
        tempState = torch.from_numpy(tempState)
        stateTensor[counter,:] = tempState
        counter+=1

    stateTensor = stateTensor.to(device)

    return  stateTensor



#stateRewards : a tensor with 'requires_grad = True'
#freq_diff : a plain tensor
def calculate_gradients(optimizer ,stateRewards , freq_diff):


    optimizer.zero_grad()
    dotProd = torch.dot(stateRewards.squeeze(), freq_diff.squeeze())
    dotProd.backward()


#************************************************************************



#takes in a costNetork and spits out a policy that is
#optimal for the particular version of costNetwork
#that has been provided to the method
#Used policy gradient method (actorCritic)
#implementation taken from the file ****ballgameActorCritic.py



#takes in the parameters of
def deepMaxEntIRL(expertDemofile,costNNparams , policyNNparams , iterations ,samplingIterations ,acGameplayiteraions):

    #initialize both the networks
    #filename = 'expertstateinfo.npy'

    #stateDict : a dictionary where key = str(numpy state array) , value : integer index
    #lookuptable : a dictionary where key : str(numpy array) , value : numpy array
    stateDict,lookputable = getstateDict('no obstacle')
    stateTensor = getStateTensor(lookputable)


    #expertFreq  = getStateVisitationFrequencyExpert(filename,stateDict) #add filename for expert demonstration
    gamePlayIterations = acGameplayiteraions
    #policyNetwork = Policy(policyNNparams)
    costNetwork = CostNetwork(costNNparams)
    optimizer = optim.Adam(costNetwork.parameters() , lr = 0.002)

    #create the game board

    #env = BE.createBoard(display =True , static_obstacle_radius= 10 , static_obstacles= 10)

    #the main IRL loop
    for i in range(iterations):

        #start with a cost function
        #start with a cost function

        #optimize policy for the provided cost function

        rlAC = ActorCritic(costNetwork = costNetwork, noofPlays= gamePlayIterations , policy_nn_params = policyNNparams)
        optimalPolicy = rlAC.actorCriticMain()

        expertFreq = rlAC.compute_state_visitation_freq_Expert(stateDict, expertDemofile)
        stateFreq = rlAC.compute_state_visitation_freq_sampling(stateDict , samplingIterations)


        print 'expert freq :',expertFreq
        print np.sum(expertFreq)
        print 'policy freq :',stateFreq
        print np.sum(stateFreq)
        #get the difference in frequency
        freq_diff = expertFreq - stateFreq
        freq_diff = torch.from_numpy(freq_diff).to(device)

        freq_diff = freq_diff.type(torch.cuda.FloatTensor)
        #calculate R for each of the state
        #takes in an array of arrays

        stateRewards = costNetwork(stateTensor)

        calculate_gradients(optimizer , stateRewards, freq_diff)

        optimizer.step()


if __name__=='__main__':

    #info = takeExpertInfo(50)

    stateDict,_ = getstateDict('no obstacle')
    #frqarray = getStateVisitationFrequencyExpert('expertstateinfolong_50.npy',stateDict)

    demofile = 'expertstateinfolong_50.npy'
    print stateDict
    #p = compute_state_visitation_freq_sampling(stateDict, 10, policy)
    nn_params ={'input': 29 , 'hidden': [3,3] , 'output':1}
    costNNparams = nn_params
    policyNNparams = nn_params
    iterations_irl = 300 #number of times the algorithm will go back and forth
    #between RL and IRL
    samplingIter = 1000 #no of samples to be played inorder to get the agent state frequency
    gameplayIter = 1000 #no of iterations in the Actor Critic method to be played
    deepMaxEntIRL(demofile, costNNparams , policyNNparams , iterations_irl , samplingIter , gameplayIter)
    '''
    costNNparams = {}
    policyNNparams = {}
    iterations = 10

    deepMaxEntIRL(costNNparams , policyNNparams , iterations)
    '''



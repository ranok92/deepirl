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
import torch.nn.utils.clip_grad as clip_grad

import numpy as np
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from collections import namedtuple
from rlmethods import ActorCritic
from networks import CostNetwork
from networks import Policy

from featureExtractor import localWindowFeature

import os
import datetime

#GLOBAL PARAMETERS

NN_PARAMS = {'input': 100,
             'hidden': [256 , 256],
             'output': 4}

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



#**have to change this part. THis is not a part of the pipeline
#but this method is called to transform raw state information from
#expert trajectories to feature vectors that will be used to calulate the
#state visitation frequency of the expert**change!!
def takeExpertInfo(trajectories,WINDOW_SIZE,gridsize,device):

    env = BE.createBoard(display=True)
    actionList = [(0,-1),(1,0),(0,1),(-1,0)]
    mainStateInfo = []

    for i in range(trajectories):

        state = env.reset()
        #state = env.get_state_BallEnv(window_size=5)
        state = localWindowFeature(state,WINDOW_SIZE,gridsize,device)
        done = False
        env.render()
        trajInfo = []
        trajInfo.append(state)
        while not done:

            action = env.take_action_from_userKeyboard()
            action = actionList[action]

            next_state , reward , done , _ = env.step(action)
            #next_state = env.get_state_BallEnv()
            next_state = localWindowFeature(state,WINDOW_SIZE,gridsize,device)
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


#returns a tensor of size nxm (where n: is the number of distinct states, m: is the size of tensor to represent a single state )
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


def toTensor(state):

    ref_state = torch.from_numpy(state)
    ref_state = ref_state.type(torch.cuda.FloatTensor)
    ref_state = ref_state.unsqueeze(0)

    return ref_state

#function taken from rlmethods.py. given a policy network and a state (of the ball environment)
#it returns an action
def select_action(state,policy):

    probs ,state_value  = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()



####################################################################################
# def checkOverlap(obj1Pos,obj1rad, obj2Pos, obj2rad):
#
#     xdiff = obj1Pos[0]-obj2Pos[0]
#     ydiff = obj1Pos[1]-obj2Pos[1]
#
#     if (np.hypot(xdiff,ydiff)-obj1rad-obj2rad) > 0:
#
#         return False
#     else:
#         return True
#
# def block_to_arrpos(window_size,x,y):
#
#     a = (window_size**2-1)/2
#     b = window_size
#     pos = a+(b*y)+x
#     return int(pos)
#
#
#
# def get_state_BallEnv(state):
#
#     #state is a list of info where 1st position holds the position of the
#     #agent, 2nd the position of the goal , 3rd the distance after that,
#     #the positions of the obstacles in the world
#         #print(state)
#     WINDOW_SIZE = 5
#     agentRad = 10
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     window_size = WINDOW_SIZE
#     block_width = 2
#
#     window_rows = window_size
#     row_start =  (window_rows-1)/2
#     window_cols = window_size
#     col_start = (window_cols-1)/2
#
#     ref_state = np.zeros(4+window_size**2)
#     #print(ref_state.shape)
#     a = (window_size**2-1)/2
#     ref_state[a+4] = 1
#     agent_pos = state[0]
#     goal_pos = state[1]
#     diff_x = goal_pos[0] - agent_pos[0]
#     diff_y = goal_pos[1] - agent_pos[1]
#     if diff_x >= 0 and diff_y >= 0:
#         ref_state[1] = 1
#     elif diff_x < 0  and diff_y >= 0:
#         ref_state[0] = 1
#     elif diff_x < 0 and diff_y < 0:
#         ref_state[3] = 1
#     else:
#         ref_state[2] = 1
#
#     for i in range(3,len(state)):
#
#         #as of now this just measures the distance from the center of the obstacle
#         #this distance has to be measured from the circumferance of the obstacle
#
#         #new method, simulate overlap for each of the neighbouring places
#         #for each of the obstacles
#         obs_pos = state[i][0:2]
#         obs_rad = state[i][2]
#         for r in range(-row_start,row_start+1,1):
#             for c in range(-col_start,col_start+1,1):
#                 #c = x and r = y
#                 temp_pos = (agent_pos[0] + c*block_width , agent_pos[1] + r*block_width)
#                 if checkOverlap(temp_pos,agentRad, obs_pos, obs_rad):
#                     pos = block_to_arrpos(window_size,r,c)
#
#                     ref_state[pos]=1
#
#     #state is as follows:
#         #first - tuple agent position
#         #second -
#     state = torch.from_numpy(ref_state).to(device)
#     state = state.type(torch.cuda.FloatTensor)
#     state = state.unsqueeze(0)
#
#     return state
#

#************************************************************************



#takes in a costNetork and spits out a policy that is
#optimal for the particular version of costNetwork
#that has been provided to the method
#Used policy gradient method (actorCritic)
#implementation taken from the file ****ballgameActorCritic.py



#function for the deepMax Entropy IRL method
class DeepMaxEntIRL:

    def __init__(self , expertDemofile , rlMethod , costNNparams , costNetworkDict , policyNNparams , policyNetworkDict , irliterations , samplingIterations , rliterations , store=False , storeInfo = None , render = False , onServer = True , resultPlotIntervals = 10 , irlModelStoreInterval = 1 , rlModelStoreInterval = 500 , testIterations = 0 , verbose=False):

        self.expertDemofile = expertDemofile
        self.rlMethod = rlMethod
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.costNNparams = costNNparams
        self.costNetwork = CostNetwork(costNNparams).to(self.device)

        self.storedCostNetwork = costNetworkDict

        if self.storedCostNetwork!=None:

	    	self.costNetwork.load_state_dict(torch.load(self.storedCostNetwork))
	    	self.costNetwork.eval()

        self.policyNNparams = policyNNparams
        self.storedPolicyNetwork = policyNetworkDict
        self.policyNetwork = None
        self.irlIterations = irliterations
        self.samplingIterations = samplingIterations
        self.rlIterations = rliterations

        self.verbose = verbose

        #parameters for display
        self.render = render
        self.onServer = onServer

        #parameters for storing results
        self.store = store
        self.storeDirsInfo = storeInfo
        self.plotIntervals = resultPlotIntervals
        self.irlModelStoreInterval = irlModelStoreInterval
        self.rlModelStoreInterval = rlModelStoreInterval
        self.testRuns = testIterations


    def compute_state_visitation_freq_Expert(self, stateDict):

        N_STATES = len(stateDict.keys())

        #trajectoryFile was created using a list of lists
        info = np.load(self.expertDemofile)
        #info is an array of size (no_of_samples_taken,)
        #for each pos of info, i.e. info[0] is a list of length : number of timesteps in that trajectory
        #for each timestep there is an array that stores the state information.
        #i.e. info[i][j] is an array describing the state information
        #print info
        no_of_samples = len(info)
        mu = np.zeros([no_of_samples, N_STATES])
        reward_array = np.zeros(no_of_samples)
        avglen = np.zeros(no_of_samples)
        #loop through each of the trajectories
        for i in range(no_of_samples):
            trajReward = 0
            for t in range(len(info[i])):
                state = info[i][t]
                stateIndex = stateDict[np.array2string(state)]
                mu[i][stateIndex]+=1
                if t!=0:

                    state_tensor = toTensor(state)
                    reward = self.costNetwork(state_tensor)
                    #print 'reward :', reward.size()
                    trajReward+=reward.item()

            reward_array[i] = np.exp(-trajReward)
            avglen[i] = t


        #print 'The reward array :',reward_array
        #print 'sum of reward_array :', np.sum(reward_array)
        #normalize the rewards array
        reward_array = np.divide(reward_array, np.sum(reward_array))
        if self.verbose: print 'Avg length of the trajectories expert:', np.dot(avglen,reward_array)
        #print 'The normalized reward array :', reward_array

        #multiply each of the trajectory state visitation freqency by their corresponding normalized reward
        #print 'state visitation freq :', mu

        for i in range(no_of_samples):

            mu[i,:] = mu[i,:]*reward_array[i]

        #print 'state visitation freq array after norm ', mu
        p =  np.sum(mu,axis=0)

        return np.expand_dims(p,axis=1)


    def runDeepMaxEntIRL(self):

        #initialize both the networks
	    #filename = 'expertstateinfo.npy'

	    #stateDict : a dictionary where key = str(numpy state array) , value : integer index
	    #lookuptable : a dictionary where key : str(numpy array) , value : numpy array
        stateDict,lookputable = getstateDict('no obstacle')
        stateTensor = getStateTensor(lookputable)


        #expertFreq  = getStateVisitationFrequencyExpert(filename,stateDict) #add filename for expert demonstration
        gamePlayIterations = self.rlIterations
        #policyNetwork = Policy(policyNNparams)

        optimizer = optim.Adam(self.costNetwork.parameters() , lr = 0.002 , weight_decay= .1 )

        #create the game board

	    #env = BE.createBoard(display =True , static_obstacle_radius= 10 , static_obstacles= 10)


	    #if storeInfo is true create stuff to store intermediate results

        if self.store:

            basePath = self.storeDirsInfo['basepath']
            curDirCost = self.storeDirsInfo['costDir']
            curDirPolicy = self.storeDirsInfo['policyDir']
            fileNameCost = self.storeDirsInfo['costFilename']
            fileNamePolicy = self.storeDirsInfo['policyFilename']

        else:

            basePath = curDirPolicy = curDirCost = fileNameCost = fileNamePolicy = None


        w = [[] for i in range(10)]
        #exit()
        xaxis = []

        #the main IRL loop
        for i in range(self.irlIterations):

        #start with a cost function
        #start with a cost function

        #optimize policy for the provided cost function
            fileNamePolicyFull = None


            if self.store:
                fileNamePolicyFull = curDirPolicy+fileNamePolicy+'iterEND_'+str(i)+'.h5'


            if self.rlMethod=='Actor_Critic':

                rlAC = ActorCritic(costNetwork = self.costNetwork, noofPlays= gamePlayIterations , policy_nn_params = self.policyNNparams, storedNetwork = self.storedPolicyNetwork,
                                 storeModels=self.store, fileName = fileNamePolicy, policyNetworkDir = curDirPolicy, basePath = basePath, irliteration = i , displayBoard = self.render , onServer = self.onServer,
                                 plotInterval = self.plotIntervals , modelSaveInterval = self.rlModelStoreInterval , verbose=  self.verbose)
                self.policyNetwork = rlAC.actorCriticMain()

            expertFreq = self.compute_state_visitation_freq_Expert(stateDict)
            stateFreq = rlAC.compute_state_visitation_freq_sampling(stateDict , self.samplingIterations)

            if self.verbose:
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

            stateRewards = self.costNetwork(stateTensor)

            calculate_gradients(optimizer , stateRewards, freq_diff)
            clip_grad.clip_grad_norm(self.costNetwork.parameters() , 100)
            optimizer.step()
            #######printing grad and weight norm##############

            if self.verbose:
                print 'Start printing grad cost network :'
                for x in self.costNetwork.parameters():
                    #print 'One'
                    print 'x cost weight: ', torch.norm(x.data)
                    if x.grad is not None:
                        print 'x cost grad ', torch.norm(x.grad)
                print 'The end.'


                print 'Start printing grad policy network :'
                for x in self.policyNetwork.parameters():
                    #print 'One'
                    print 'x cost weight: ', torch.norm(x.data)
                    if x.grad is not None:
                        print 'x cost grad ', torch.norm(x.grad)
                print 'The end.'
            #####################plotting the weight norms#######################
            if self.store:
                if i % self.irlModelStoreInterval==0:
                    torch.save(self.costNetwork.state_dict(),curDirCost+fileNameCost+'iteration_'+str(i)+'.h5')
                    torch.save(self.policyNetwork.state_dict(),fileNamePolicyFull)



    def testMaxDeepIRL(self):
        '''
        this is a method to test a model
        runIterations
        environment instatiation information?
            size of the environment
            number of obstacles
            agent radius
            window size for state transformation (this should match with the
                parameters of the policynetwork model being used for the run)

        Given the above information, this method shows the performance of the current model in the
        provided environment
        '''
        actionList = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,0),(-1,-1)]

        optimizer = optim.Adam(self.costNetwork.parameters() , lr = 0.002 , weight_decay= .1 )

        #intialize the policyNetwork

        self.policyNetwork = Policy(self.policyNNparams).to(self.device)
        self.policyNetwork.load_state_dict(torch.load(self.storedPolicyNetwork))
        self.policyNetwork.eval()

        #intialize the test environment
        #get the board information from the user and store it in a dictionary.
        #use it here
        RUNLIMIT = 400 #this should also be passed as a parameter
        env = BE.createBoard(display= self.render)
        rewardAcrossRun = []
        xListAcrossRun = []

        plt.figure(1)
        plt.title('Plotting rewards across multiple runs:')

        ##
        WINDOW_SIZE = 5
        GRID_SIZE = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ##

        print 'Number of runs to be done :', self.testRuns
        for run_i in range(self.testRuns):

            state = env.reset()

            state = localWindowFeature(state,WINDOW_SIZE,GRID_SIZE,device) #convert the state to usabe state information array

            rewardPerRun = []
            xListPerRun = []
            done = False
            runcounter = 0
            totalReward = 0
            #provision to plot the reward per run and plot across multiple runs
            plt.figure(2)
            plt.title('Plotting rewards from a single run: {}'.format(run_i))

            while runcounter <= RUNLIMIT:

                runcounter+=1
                actionIndex = select_action(state, self.policyNetwork)
                action = actionList[actionIndex]

                nextState , reward, done , _ = env.step(action)

                nextState = localWindowFeature(nextState,WINDOW_SIZE,GRID_SIZE,device)
                reward = self.costNetwork(nextState)

                totalReward += reward

                if self.render:
                    env.render()

                if done:
                    print 'done and dusted'
                    break


                xListPerRun.append(runcounter)
                rewardPerRun.append(reward)
                plt.plot(xListPerRun , rewardPerRun , color = 'blue')
                plt.draw()
                plt.pause(.0001)

            xListAcrossRun.append(run_i)
            rewardAcrossRun.append(totalReward)

            plt.plot(xListAcrossRun , rewardAcrossRun , color='black')
            plt.draw()
            plt.pause(.0001)

        return 0

if __name__=='__main__':

    #info = takeExpertInfo(50)

    #stateDict,_ = getstateDict('no obstacle')
    #frqarray = getStateVisitationFrequencyExpert('expertstateinfolong_50.npy',stateDict)

    demofile = 'expertstateinfolong_50.npy'
    info = np.load(demofile)

    print info
    '''
    #p = compute_state_visitation_freq_sampling(stateDict, 10, policy)
    nn_params ={'input': 29 , 'hidden': [256,128] , 'output':4}
    costNNparams = nn_params
    policyNNparams = nn_params
    iterations_irl = 30 #number of times the algorithm will go back and forth
    #between RL and IRL
    samplingIter = 2000 #no of samples to be played inorder to get the agent state frequency
    gameplayIter = 1000 #no of iterations in the Actor Critic method to be played
    deepMaxEntIRL(demofile, costNNparams , policyNNparams , iterations_irl , samplingIter , gameplayIter ,storeInfo=True)
    
    costNNparams = {}
    policyNNparams = {}
    iterations = 10

    deepMaxEntIRL(costNNparams , policyNNparams , iterations)
    '''



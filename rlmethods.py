import datetime
import os
from collections import namedtuple

import ballenv_pygame as BE
import matplotlib
import numpy as np
import torch
import torch.cuda as cutorch
import torch.nn.functional as F
import torch.nn.utils.clip_grad as clip_grad
import torch.optim as optim
from featureExtractor import localWindowFeature
from matplotlib import pyplot as plt
from torch.distributions import Categorical

from networks import CostNetwork, Policy


def getMemoryAllocationInfo(memoryInBytes):

    result = ''
    val = memoryInBytes
    infoList = [0, 0, 0, 0]
    i = 0
    while val > 0:

        infoList[i] = val % 1024
        val /= 1024
        i += 1

    result = '{} Gb, {} Mb, {} Kb ,{} b'.format(
        infoList[3], infoList[2], infoList[1], infoList[0])

    return result, infoList


class HistoryBuffer():

    def __init__(self, bufferSize=10):

        self.bufferSize = bufferSize
        self.buffer = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    # add a state to the history buffer
    # each state is assumed to be of shape ( 1 x S )

    def addState(self, state):

        if len(self.buffer) >= self.bufferSize:

            del self.buffer[0]  # remove the oldest state
        self.buffer.append(state.cpu().numpy())

    # returns the 10 states in the buffer in the form of a torch tensor in the order in which they
    # were encountered
    def getHistory(self):

        arrSize = self.buffer[0].shape[1]
        #print 'ArraySize',arrSize
        arrayHist = np.asarray(self.buffer)

        arrayHist = np.reshape(arrayHist, (1, arrSize*self.bufferSize))
        state = torch.from_numpy(arrayHist).to(self.device)
        state = state.type(torch.cuda.FloatTensor)
        #state = state.unsqueeze(0)

        return state


class ActorCritic:

    def __init__(self, costNetwork=None, noofPlays=100, policy_nn_params={}, storedNetwork=None, Gamma=.9, Eps=.00001, storeModels=True,
                 fileName=None, basePath=None, policyNetworkDir=None, plotInterval=10, irliteration=None, displayBoard=False, onServer=True,
                 modelSaveInterval=500, verbose=False):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.irlIter = irliteration
        self.storedPolicyNetwork = storedNetwork

        self.policy = Policy(policy_nn_params).to(self.device)
        self.verbose = verbose
        if self.storedPolicyNetwork != None:

            self.policy.load_state_dict(torch.load(self.storedPolicyNetwork))
            self.policy.eval()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.gamma = Gamma
        self.eps = Eps
        self.costNet = costNetwork.to(self.device)
        self.no_of_plays = noofPlays

        self.displayBoard = displayBoard

        self.onServer = onServer

        self.env = BE.createBoard(display=self.displayBoard,
                                  static_obstacles=0,
                                  static_obstacle_radius=10)

        self.WINDOW_SIZE = 5
        self.agentRad = 10
        self.avgReturn = 0

        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.StoreModels = storeModels
        self.logInterval = modelSaveInterval
        self.plotInterval = plotInterval
        self.move_list = [(1, 1), (1, -1), (1, 0), (0, 1),
                          (0, -1), (0, 0), (-1, 1), (-1, 0), (-1, -1)]
        self.basePath = basePath
        self.fileName = fileName
        self.curDirPolicy = policyNetworkDir

    def agent_action_to_WorldActionSimplified(self, action):
        if action == 0:  # move front
            return np.asarray([0, -5])
        if action == 1:  # move right
            return np.asarray([5, 0])
        if action == 2:  # move down
            return np.asarray([0, 5])
        if action == 3:  # move left
            return np.asarray([-5, 0])

    def select_action(self, state, policy):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        '''
        print 'Start printing grad :'
        for x in policy.parameters():
            #print 'One'
            print 'x weight: ', torch.norm(x.data)
            if x.grad is not None:
                print 'x grad ', torch.norm(x.grad)
        print 'The end.'
        '''
        probs, state_value = policy(state)
        #print 'probs :' , probs
        m = Categorical(probs)
        action = m.sample()

        policy.saved_actions.append(
            self.SavedAction(m.log_prob(action), state_value))
        return action.item()

    # stateDict - OrderedDict : key : state(string) , value : integer
    # trajs - number of trajectories to simulate
    # policy network

    def toTensor(self, state):

        ref_state = torch.from_numpy(state).to(self.device)
        ref_state = ref_state.type(torch.cuda.FloatTensor)
        ref_state = ref_state.unsqueeze(0)

        return ref_state

    def compute_state_visitation_freq_Expert(self, stateDict, trajectoryFile):

        N_STATES = len(stateDict.keys())

        # trajectoryFile was created using a list of lists
        info = np.load(trajectoryFile)
        # info is an array of size (no_of_samples_taken,)
        # for each pos of info, i.e. info[0] is a list of length : number of timesteps in that trajectory
        # for each timestep there is an array that stores the state information.
        # i.e. info[i][j] is an array describing the state information
        #print info
        no_of_samples = len(info)
        mu = np.zeros([no_of_samples, N_STATES])
        reward_array = np.zeros(no_of_samples)
        avglen = np.zeros(no_of_samples)
        # loop through each of the trajectories
        for i in range(no_of_samples):
            trajReward = 0
            for t in range(len(info[i])):
                state = info[i][t]
                stateIndex = stateDict[np.array2string(state)]
                mu[i][stateIndex] += 1
                if t != 0:

                    state_tensor = self.toTensor(state)
                    reward = self.costNet(state_tensor)
                    #print 'reward :', reward.size()
                    trajReward += reward.item()

            reward_array[i] = np.exp(-trajReward)
            avglen[i] = t

        #print 'The reward array :',reward_array
        #print 'sum of reward_array :', np.sum(reward_array)
        # normalize the rewards array
        reward_array = np.divide(reward_array, np.sum(reward_array))
        if self.vebose:
            print 'Avg length of the trajectories expert:', np.dot(
                avglen, reward_array)
        #print 'The normalized reward array :', reward_array

        # multiply each of the trajectory state visitation freqency by their corresponding normalized reward
        #print 'state visitation freq :', mu

        for i in range(no_of_samples):

            mu[i, :] = mu[i, :]*reward_array[i]

        #print 'state visitation freq array after norm ', mu
        p = np.sum(mu, axis=0)

        return np.expand_dims(p, axis=1)

    # calculates the state visitation frequency of an agent
    # stateDict : a dictionary where key = str(numpy state array) , value : integer index
    # lookuptable : a dictionary where key : str(numpy array) , value : numpy array
    def compute_state_visitation_freq_sampling(self, stateDict, no_of_trajs):

        N_STATES = len(stateDict.keys())
        N_ACTIONS = 4

        no_of_samples = no_of_trajs
        '''

        run a bunch of trajectories, get the cost for each of them c_theta(tao)
        prob of a trajectory is directly proportional to the cost it obtains exp(-c_theta(tao)
        multiply the prob with the state visitation for each of the trajectory
        update Z (the normalizing factor)
        '''
        T = 200
        # mu[s, t] is the prob of visiting state s at time t
        mu = np.zeros([no_of_samples, N_STATES])

        # get the start states
        avglen = np.zeros(no_of_samples)
        reward_array = np.zeros(no_of_samples)

        for i in range(no_of_samples):

            # reset returns the original state info , but here we need the local 29 x 1 vector
            state = self.env.reset()
            state = localWindowFeature(
                state, self.WINDOW_SIZE, 2, self.device).squeeze().cpu().numpy()

            stateIndex = stateDict[np.array2string(state)]
            mu[i][stateIndex] += 1
            done = False
            traj_reward = 0
            # running for a single trajectory
            for t in range(1, T):

                state = self.toTensor(state)
                action = self.select_action(state, self.policy)
                action = self.agent_action_to_WorldActionSimplified(action)

                next_state, reward, done, _ = self.env.step(action)
                # ******IMP**** state returned from env.step() is different from the state representation being used for the
                # networks
                next_state = localWindowFeature(
                    next_state, self.WINDOW_SIZE, 2, self.device).squeeze().cpu().numpy()

                next_state_Index = stateDict[np.array2string(next_state)]
                #print 'type of next state', next_state.dtype

                next_state_tensor = self.toTensor(next_state)
                reward = self.costNet(next_state_tensor)
                traj_reward += reward.item()  # keep adding the rewards obtained in each state

                mu[i][next_state_Index] += 1
                state = next_state

                if done:
                    break

            # the literature suggests exp(-C(traj)) where C(traj) is the cost of the trajectory
            reward_array[i] = np.exp(-traj_reward)
            # as because we are dealing with rewards, so I removed the negative sign
            avglen[i] = t

        if self.verbose:
            print 'traj reward :', traj_reward
            print 'The reward array :', reward_array

        # normalize the rewards array

        reward_array = np.divide(reward_array, sum(reward_array))

        if self.verbose:
            print 'Avg length of the trajectories :', np.dot(
                avglen, reward_array)
            print 'The normalized reward array :', reward_array

        # multiply each of the trajectory state visitation freqency by their corresponding normalized reward
        #print 'state visitation freq :', mu

        for i in range(no_of_samples):

            mu[i, :] = mu[i, :]*reward_array[i]

        #print 'state visitation freq array after norm ', mu
        p = np.sum(mu, axis=0)

        return np.expand_dims(p, axis=1)
        '''
        print 'Avg length for agent sampling :', avglen/no_of_samples
        print 'State visitation freq :',mu[:,0],'Sum :',sum(mu[:,0])
        for t in range(1,T):

            mu[:,t] = np.divide(mu[:,t],no_of_samples)

        p = np.sum(mu,1)
        #p = np.divide(p,no_of_samples)
        p = np.expand_dims(p,axis=1)
        return p
        '''

# the code for actor_critic is taken from here :
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
    def finish_episode(self):

        if self.verbose:
            print 'Inside finish episode :'

        R = 0
        saved_actions = self.policy.saved_actions
        policy_losses = []
        value_losses = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        if self.verbose:
            print 'rewards :', rewards
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            #print value.shape
            #print torch.tensor([r]).to(device).shape
            value_losses.append(F.smooth_l1_loss(
                value, torch.tensor([r]).to(self.device).unsqueeze(0)))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()
        loss.backward()
        clip_grad.clip_grad_norm(self.policy.parameters(), 100)
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]
        return loss

    def actorCriticMain(self):

        historySize = 1
        hbuffer = HistoryBuffer(historySize)
        # actorCriticWindow-windowsize - state obtained from local window
        # actorCriticFeaures - state obtained from features
        # actirCriticFeaturesFull - state obtained from using all features
        # actorCriticXXXHistory  - state obtained from any of the above methods and using a history buffer

        if self.StoreModels:

            filename = 'actorCriticWindow5History'
            curDay = str(datetime.datetime.now().date())
            curtime = str(datetime.datetime.now().time())
            if self.basePath is None:
                self.basePath = 'saved-models_trainBlock' + '/evaluatedPoliciesTest/'
            subPath = curDay + '/' + curtime + '/'
            curDir = self.basePath + subPath
            plotDir = 'saved_plots/actorCritic/'

            # if filename==None:
            #    os.makedirs(curDir)

            if self.basePath is not None:
                os.makedirs(self.basePath+'ploting_'+str(self.irlIter))
            # if os.path.exists(curDir):
            #    print "YES"

        # ******************************

        state = self.env.reset()
        rewardList = []
        lossList = []
        nnRewardList = []
        runList = []
        timeList = []
        #fig = plt.figure(1)
        #lossFig = plt.figure(2)
        plt.clf()
        for i_episode in range(self.no_of_plays):
            running_reward = self.eps
            state = self.env.reset()
            # env.render()
            print 'Starting episode :', i_episode
            result, infoList = getMemoryAllocationInfo(
                torch.cuda.memory_allocated(0))
            print 'Current memory usage :', result

            if infoList[2] > 100:
                print 'Clearing cache :'
                torch.cuda.empty_cache()
                result, infoList = getMemoryAllocationInfo(
                    torch.cuda.memory_allocated(0))
                print 'Memory usage after clearing cache:', result
            state = localWindowFeature(state, 5, 2, self.device)
            hbuffer.addState(state)
            #state = hbuffer.getHistory()
            #state = env.sensor_readings
            rewardPerRun = 0
            for t in range(500):  # Don't create infinite loop while learning

                if t <= historySize:

                    action = np.random.randint(0, 9)
                    action = self.move_list[action]
                    state, reward, done, _ = self.env.step(action)

                    state = localWindowFeature(
                        state, self.WINDOW_SIZE, 2, self.device)
                    reward = self.costNet(state)
                    hbuffer.addState(state)
                else:
                    state = hbuffer.getHistory()
                    action = self.select_action(state, self.policy)
                    #print action
                    if action != None:
                        action = self.move_list[action]
                        #action = agent_action_to_WorldActionSimplified(action)
                        #print action
                        state, reward, done, _ = self.env.step(action)

                        state = localWindowFeature(
                            state, self.WINDOW_SIZE, 2, self.device)

                        reward = self.costNet(state)
                        rewardPerRun += reward
                        #state = env.sensor_readings
                        hbuffer.addState(state)
                        #state = hbuffer.getHistory()
                        if i_episode % self.logInterval == 0:
                            if self.displayBoard:
                                if self.verbose:
                                    print 'ssss'
                                self.env.render()
                        self.policy.rewards.append(reward)
                        if done:
                            #print done
                            break
                        running_reward += reward
                    else:
                        continue
                # if t%500==0:
                    #print "T :",t
            #running_reward = running_reward * 0.99 + t * 0.01
            nnRewardList.append(rewardPerRun)
            rewardList.append(self.env.total_reward_accumulated)
            runList.append(i_episode)
            timeList.append(float(t)/500)
            #plt.plot(runList, rewardList,color='black')
            #plt.plot(runList , timeList , color= 'red')
            plt.figure(1)
            plt.title('Plotting the Rewards :')
            plt.plot(runList, nnRewardList, color='blue')
            plt.draw()
            plt.pause(.0001)

            if self.StoreModels:

                if i_episode % self.plotInterval == 0:
                    if self.basePath != None:
                        plt.savefig(
                            self.basePath+'ploting_'+str(self.irlIter)+'/Rewards_plotNo{}'.format(i_episode))
                #print 'The running reward for episode {}:'.format(i_episode),running_reward
                if i_episode % self.logInterval == 0:
                    if self.fileName != None:
                        torch.save(self.policy.state_dict(
                        ), self.curDirPolicy+self.fileName+str(self.irlIter)+'-'+str(i_episode)+'.h5')
                        #torch.save(self.policy.state_dict(),'saved-models_'+ 'trainBlock' +'/evaluatedPoliciesTest/'+subPath+str(i_episode)+'-'+ filename + '-' + str(i_episode) + '.h5', )
                    # else:
                    #    print 'Storing from here :'
                    #    torch.save(self.policy.state_dict(),self.filename)

                # save the model
            lossList.append(self.finish_episode())
            plt.figure(2)
            plt.title('Plotting the loss :')
            plt.plot(runList, lossList, color='red')
            plt.draw()
            plt.pause(.0001)
            if self.StoreModels:
                if i_episode % self.plotInterval == 0:
                    if self.basePath != None:
                        plt.savefig(
                            self.basePath+'ploting_'+str(self.irlIter)+'/Loss_plotNo{}'.format(i_episode))

            # plt.show()
            # if i_episode+1 % log_interval == 0:
            #    print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            #        i_episode, t, running_reward))
                # env.render()

        return self.policy


if __name__ == '__main__':

    '''
    cNN  = {'input':29 , 'hidden': [512 , 128] , 'output':1}
    pNN = {'input':29 , 'hidden': [512 , 128] , 'output':9}
    costNetwork = CostNetwork(cNN)
    rlAC = ActorCritic(costNetwork=costNetwork , policy_nn_params= pNN ,  noofPlays = 100, Gamma = .9 , Eps = .00001 , storeModels = False , loginterval = 10 , plotinterval = 2)
    p = rlAC.actorCriticMain()
    '''
    net = torch.nn.Linear(30000, 1).cuda()
    data = torch.ones(10, 30000).cuda()
    for rep in range(100000):
        batch = torch.autograd.Variable(data, requires_grad=True)
        net(batch).norm(2).backward(create_graph=True)
        results, _ = getMemoryAllocationInfo(torch.cuda.memory_allocated(0))
        print 'Memory usage : ', results

    cNN = {'input': 29, 'hidden': [512, 128], 'output': 1}
    pNN = {'input': 29, 'hidden': [512, 128], 'output': 9}
    costNetwork = CostNetwork(cNN)
    rlAC = ActorCritic(costNetwork=costNetwork, policy_nn_params=pNN,  noofPlays=100,
                       Gamma=.9, Eps=.00001, storeModels=False, loginterval=10, plotinterval=2)
    p = rlAC.actorCriticMain()

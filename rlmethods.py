import torch
import torch.nn.functional as F
import torch.optim as optim

from networks import Policy
from networks import CostNetwork

import numpy as np
import ballenv_pygame as BE
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from collections import namedtuple
import datetime
import os


class HistoryBuffer():

    def __init__(self,bufferSize = 10):

        self.bufferSize = bufferSize
        self.buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #add a state to the history buffer
    #each state is assumed to be of shape ( 1 x S )
    def addState(self , state):

        if len(self.buffer) >= self.bufferSize:

            del self.buffer[0] #remove the oldest state
        self.buffer.append(state.cpu().numpy())


    #returns the 10 states in the buffer in the form of a torch tensor in the order in which they
    #were encountered
    def getHistory(self):

        arrSize = self.buffer[0].shape[1]
        #print 'ArraySize',arrSize
        arrayHist = np.asarray(self.buffer)

        arrayHist = np.reshape(arrayHist , (1,arrSize*self.bufferSize))
        state = torch.from_numpy(arrayHist).to(self.device)
        state = state.type(torch.cuda.FloatTensor)
        #state = state.unsqueeze(0)

        return state



class ActorCritic:


    def __init__(self ,costNetwork = None , noofPlays = 100 , policy_nn_params = {} , Gamma = .9 , Eps = .00001 , storeModels = False , loginterval = 10 , plotinterval = 2):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = Policy(policy_nn_params).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.gamma = Gamma
        self.eps = Eps
        self.costNet = costNetwork.to(self.device)
        self.no_of_plays = noofPlays
        self.displayBoard = False
        self.env = BE.createBoard(display = self.displayBoard , static_obstacles= 0 , static_obstacle_radius= 10)
        self.WINDOW_SIZE = 5
        self.agentRad = 10
        self.avgReturn = 0
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.StoreModels = storeModels
        self.logInterval = loginterval
        self.plotInterval = plotinterval
        self.move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,0),(-1,-1)]

    def block_to_arrpos(self,window_size,x,y):

        a = (window_size**2-1)/2
        b = window_size
        pos = a+(b*y)+x
        return int(pos)


    def get_state_BallEnv(self,state):

    #state is a list of info where 1st position holds the position of the
    #agent, 2nd the position of the goal , 3rd the distance after that,
    #the positions of the obstacles in the world
        #print(state)
        window_size = self.WINDOW_SIZE
        block_width = 2

        window_rows = window_size
        row_start =  (window_rows-1)/2
        window_cols = window_size
        col_start = (window_cols-1)/2

        ref_state = np.zeros(4+window_size**2)
        #print(ref_state.shape)
        a = (window_size**2-1)/2
        ref_state[a+4] = 1
        agent_pos = state[0]
        goal_pos = state[1]
        diff_x = goal_pos[0] - agent_pos[0]
        diff_y = goal_pos[1] - agent_pos[1]
        if diff_x >= 0 and diff_y >= 0:
            ref_state[1] = 1
        elif diff_x < 0  and diff_y >= 0:
            ref_state[0] = 1
        elif diff_x < 0 and diff_y < 0:
            ref_state[3] = 1
        else:
            ref_state[2] = 1

        for i in range(3,len(state)):

            #as of now this just measures the distance from the center of the obstacle
            #this distance has to be measured from the circumferance of the obstacle

            #new method, simulate overlap for each of the neighbouring places
            #for each of the obstacles
            obs_pos = state[i][0:2]
            obs_rad = state[i][2]
            for r in range(-row_start,row_start+1,1):
                for c in range(-col_start,col_start+1,1):
                    #c = x and r = y
                    temp_pos = (agent_pos[0] + c*block_width , agent_pos[1] + r*block_width)
                    if self.checkOverlap(temp_pos,self.agentRad, obs_pos, obs_rad):
                        pos = self.block_to_arrpos(window_size,r,c)

                        ref_state[pos]=1

        #state is as follows:
            #first - tuple agent position
            #second -
        state = torch.from_numpy(ref_state).to(self.device)
        state = state.type(torch.cuda.FloatTensor)
        state = state.unsqueeze(0)

        return state


    #returns true if there is an overlap
    def checkOverlap(self,obj1Pos,obj1rad, obj2Pos, obj2rad):

        xdiff = obj1Pos[0]-obj2Pos[0]
        ydiff = obj1Pos[1]-obj2Pos[1]

        if (np.hypot(xdiff,ydiff)-obj1rad-obj2rad) > 0:

            return False
        else:
            return True

    def agent_action_to_WorldActionSimplified(self,action):
        if action==0: #move front
            return np.asarray([0,-5])
        if action==1: #move right
            return np.asarray([5,0])
        if action==2: #move down
            return np.asarray([0,5])
        if action==3: #move left
            return np.asarray([-5,0])


    def select_action(self,state,policy):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        '''
        for x in policy.parameters():
            #print 'One'
            #print 'x : ', torch.norm(x.data)
            if x.grad is not None:
                print 'x grad ', torch.norm(x.grad)
        print 'The state :',state
        '''
        probs ,state_value  = policy(state)
        #print 'probs :' , probs
        m = Categorical(probs)
        action = m.sample()
        #print action

        policy.saved_actions.append(self.SavedAction(m.log_prob(action), state_value))
        return action.item()

    #stateDict - OrderedDict : key : state(string) , value : integer
    #trajs - number of trajectories to simulate
    #policy network

    def toTensor(self,state):

        ref_state = torch.from_numpy(state).to(self.device)
        ref_state = ref_state.type(torch.cuda.FloatTensor)
        ref_state = ref_state.unsqueeze(0)

        return ref_state


    def compute_state_visitation_freq_Expert(self, stateDict , trajectoryFile):

        N_STATES = len(stateDict.keys())

        #trajectoryFile was created using a list of lists
        info = np.load(trajectoryFile)
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

                    state_tensor = self.toTensor(state)
                    reward = self.costNet(state_tensor)
                    #print 'reward :', reward.size()
                    trajReward+=reward.item()

            reward_array[i] = np.exp(trajReward)
            avglen[i] = t


        #print 'The reward array :',reward_array
        #print 'sum of reward_array :', np.sum(reward_array)
        #normalize the rewards array
        reward_array = np.divide(reward_array, np.sum(reward_array))
        print 'Avg length of the trajectories expert:', np.dot(avglen,reward_array)
        #print 'The normalized reward array :', reward_array

        #multiply each of the trajectory state visitation freqency by their corresponding normalized reward
        #print 'state visitation freq :', mu

        for i in range(no_of_samples):

            mu[i,:] = mu[i,:]*reward_array[i]

        #print 'state visitation freq array after norm ', mu
        p =  np.sum(mu,axis=0)

        return np.expand_dims(p,axis=1)




    #calculates the state visitation frequency of an agent
    #stateDict : a dictionary where key = str(numpy state array) , value : integer index
    #lookuptable : a dictionary where key : str(numpy array) , value : numpy array
    def compute_state_visitation_freq_sampling(self,stateDict, no_of_trajs):


        N_STATES = len(stateDict.keys())
        N_ACTIONS = 4
        print 'P_A_shape'

        no_of_samples = no_of_trajs
        '''
      
        run a bunch of trajectories, get the cost for each of them c_theta(tao)
        prob of a trajectory is directly proportional to the cost it obtains exp(-c_theta(tao)
        multiply the prob with the state visitation for each of the trajectory
        update Z (the normalizing factor)
        '''
        T = 400
        # mu[s, t] is the prob of visiting state s at time t
        mu = np.zeros([no_of_samples, N_STATES])

        # get the start states
        avglen = np.zeros(no_of_samples)
        reward_array = np.zeros(no_of_samples)

        for i in range(no_of_samples):

            state = self.env.reset() #reset returns the original state info , but here we need the local 29 x 1 vector
            state = self.env.get_state_BallEnv(window_size=5)

            stateIndex = stateDict[np.array2string(state)]
            mu[i][stateIndex]+=1
            done = False
            traj_reward = 0
            #running for a single trajectory
            for t in range(1,T):

                state = self.toTensor(state)
                action = self.select_action(state,self.policy)
                action = self.agent_action_to_WorldActionSimplified(action)
                next_state ,reward , done, _ = self.env.step(action)
                #******IMP**** state returned from env.step() is different from the state representation being used for the
                #networks
                next_state = self.env.get_state_BallEnv(window_size=5)
                next_state_Index = stateDict[np.array2string(next_state)]
                #print 'type of next state', next_state.dtype

                next_state_tensor = self.toTensor(next_state)
                reward = self.costNet(next_state_tensor)
                traj_reward+=reward.item() #keep adding the rewards obtained in each state


                mu[i][next_state_Index]+=1
                state = next_state

                if done:
                    break

            reward_array[i] = np.exp(traj_reward) #the literature suggests exp(-C(traj)) where C(traj) is the cost of the trajectory
                                                     #as because we are dealing with rewards, so I removed the negative sign
            avglen[i] = t


        print 'The reward array :',reward_array

        #normalize the rewards array
        reward_array = np.divide(reward_array, sum(reward_array))
        print 'Avg length of the trajectories :', np.dot(avglen,reward_array)
        print 'The normalized reward array :', reward_array

        #multiply each of the trajectory state visitation freqency by their corresponding normalized reward
        print 'state visitation freq :', mu

        for i in range(no_of_samples):

            mu[i,:] = mu[i,:]*reward_array[i]

        print 'state visitation freq array after norm ', mu
        p =  np.sum(mu,axis=0)

        return np.expand_dims(p,axis=1)
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


    def finish_episode(self):

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
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            #print value.shape
            #print torch.tensor([r]).to(device).shape
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]).to(self.device).unsqueeze(0)))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]
        return loss



    def actorCriticMain(self):

        historySize = 1
        hbuffer = HistoryBuffer(historySize)
        #actorCriticWindow-windowsize - state obtained from local window
        #actorCriticFeaures - state obtained from features
        #actirCriticFeaturesFull - state obtained from using all features
        #actorCriticXXXHistory  - state obtained from any of the above methods and using a history buffer

        if self.StoreModels:
            filename = 'actorCriticWindow5History'
            curDay = str(datetime.datetime.now().date())
            curtime = str(datetime.datetime.now().time())
            basePath = 'saved-models_trainBlock' +'/evaluatedPoliciesTest/'
            subPath = curDay + '/' + curtime + '/'
            curDir = basePath + subPath
            os.makedirs(curDir)
            if os.path.exists(curDir):
                print "YES"

        #******************************


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
            #env.render()
            print 'Starting episode :', i_episode
            state = self.get_state_BallEnv(state)
            hbuffer.addState(state)
            #state = hbuffer.getHistory()
            #state = env.sensor_readings
            rewardPerRun = 0
            for t in range(500):  # Don't create infinite loop while learning

                if t <= historySize:

                    action = np.random.randint(0,9)
                    action = self.move_list[action]
                    state, reward , done , _ = self.env.step(action)


                    state = self.get_state_BallEnv(state)
                    reward = self.costNet(state)
                    hbuffer.addState(state)
                else:
                    state = hbuffer.getHistory()
                    action = self.select_action(state,self.policy)
                    #print action
                    if action!=None:
                        action = self.move_list[action]
                        #action = agent_action_to_WorldActionSimplified(action)
                        #print action
                        state, reward, done, _ = self.env.step(action)

                        state = self.get_state_BallEnv(state)

                        reward = self.costNet(state)
                        rewardPerRun+=reward
                        #state = env.sensor_readings
                        hbuffer.addState(state)
                        #state = hbuffer.getHistory()
                        if i_episode%self.logInterval==0:
                            if self.displayBoard:
                                self.env.render()
                        self.policy.rewards.append(reward)
                        if done:
                            #print done
                            break
                        running_reward += reward
                    else:
                        continue
                #if t%500==0:
                    #print "T :",t
            #running_reward = running_reward * 0.99 + t * 0.01
            nnRewardList.append(rewardPerRun)
            rewardList.append(self.env.total_reward_accumulated)
            runList.append(i_episode)
            timeList.append(float(t)/500)
            #plt.plot(runList, rewardList,color='black')
            #plt.plot(runList , timeList , color= 'red')
            plt.figure(1)
            plt.plot(runList , nnRewardList , color = 'blue')
            plt.draw()
            plt.pause(.0001)

            if self.StoreModels:
                if i_episode%self.plotInterval==0:
                    plt.savefig('saved_plots/actorCritic/plotNo{}'.format(i_episode))
                #print 'The running reward for episode {}:'.format(i_episode),running_reward
                if i_episode%self.logInterval==0:
                    torch.save(self.policy.state_dict(),'saved-models_'+ 'trainBlock' +'/evaluatedPoliciesTest/'+subPath+str(i_episode)+'-'+ filename + '-' + str(i_episode) + '.h5', )

                #save the model
            lossList.append(self.finish_episode())
            plt.figure(2)
            plt.plot(runList , lossList , color = 'red')
            plt.draw()
            plt.pause(.0001)
            #plt.show()
            #if i_episode+1 % log_interval == 0:
            #    print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            #        i_episode, t, running_reward))
                #env.render()


        return self.policy

if __name__=='__main__':

    cNN  = {'input':29 , 'hidden': [512 , 128] , 'output':1}
    pNN = {'input':29 , 'hidden': [512 , 128] , 'output':9}
    costNetwork = CostNetwork(cNN)
    rlAC = ActorCritic(costNetwork=costNetwork , policy_nn_params= pNN ,  noofPlays = 100, Gamma = .9 , Eps = .00001 , storeModels = False , loginterval = 10 , plotinterval = 2)
    p = rlAC.actorCriticMain()

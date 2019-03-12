import pygame
import numpy as np
import torch
import time
import pdb

import sys
sys.path.insert(0, '..')
import utils  # NOQA: E402

class MockActionspace:
    def __init__(self, n):
        self.n = n

class MockSpec:
    def __init__(self, reward_threshold):
        self.reward_threshold = reward_threshold

class GridWorldClockless:

    #the numbering starts from 0,0 from topleft corner and goes down and right
    #the obstacles should be a list of 2 dim numpy array stating the position of the 
    #obstacle
    def __init__(
        self,
        seed = 7,
        rows = 10,
        cols = 10,
        width = 10,
        goal_state = None,
        obstacles = None,
        display = True,
        stepReward=0.001,
        step_wrapper=utils.identity_wrapper,
        reset_wrapper=utils.identity_wrapper,
    ):

        #environment information
        np.random.seed(seed)
        pygame.init()
        #pygame.key.set_repeat(1,200)
        self.rows = rows
        self.cols = cols
        self.cellWidth = width
        self.upperLimit = np.asarray([self.rows-1, self.cols-1])
        self.lowerLimit = np.asarray([0,0])
        self.agent_state = np.asarray([np.random.randint(0,self.rows-1),np.random.randint(0,self.cols-1)])
        self.state = self.onehotrep()

        # these wrappers ensure correct output format
        self.step_wrapper = step_wrapper
        self.reset_wrapper = reset_wrapper

        if goal_state==None:
            self.goal_state = np.asarray([np.random.randint(0,self.rows-1),np.random.randint(0,self.cols-1)])
        else:
            self.goal_state = goal_state

        #using manhattan distance
        self.distanceFromgoal = np.sum(np.abs(self.agent_state-self.goal_state))


        self.display = display
        self.gameDisplay = None
        self.gameExit = False

        #some colors for the display
        self.white = (255,255,255)
        self.black = (0,0,0)
        self.green = (0,255,0)
        self.red = (255,0,0)

        self.agent_action_keyboard = [False for i in range(4)]
        #does not matter if none or not.
        self.obstacles = obstacles
        # 0: up, 1: right, 2: down, 3: left
        self.actionArray = [np.asarray([-1,0]),np.asarray([0,1]),np.asarray([1,0]),
                            np.asarray([0,-1]),np.asarray([0,0])]
        self.stepReward = stepReward

        # TODO: Remove the below mock environment in favor of gym.space
        # creates a mock object mimicking action_space to obtain number of
        # actions

        self.action_space = MockActionspace(len(self.actionArray))

        # TODO: Remove the below mock spec in favor of gym style spec
        # creates an environment spec containing useful info, notably reward
        # threshold at which the env is considered to be solved


        self.spec = MockSpec(1.0)


    def reset(self):

        self.agent_state = np.asarray([np.random.randint(0,self.rows-1),np.random.randint(0,self.cols-1)])
        self.distanceFromgoal = np.sum(np.abs(self.agent_state-self.goal_state))
        self.state = self.onehotrep()
        if self.display:
            self.gameDisplay = pygame.display.set_mode((self.cols*self.cellWidth,self.rows*self.cellWidth))
            pygame.display.set_caption('Your friendly grid environment')
            self.render()

        self.state = self.reset_wrapper(self.state)
        return self.state

    #action is a number which points to the index of the action to be taken
    def step(self,action):
        #print('printing the keypress status',self.agent_action_keyboard)
        self.agent_state = np.maximum(np.minimum(self.agent_state+self.actionArray[action],self.upperLimit),self.lowerLimit)
        reward, done = self.calculateReward()
        if self.display:
            self.render()

        # step should return fourth element 'info'
        self.state = self.onehotrep()


        self.state, reward, done, _ = self.step_wrapper(
            self.state,
            reward,
            done,
            None
        )

        return self.state, reward, done, None



    def calculateReward(self):

        hit = False
        done = False

        if self.obstacles is not None:
            for obs in self.obstacles:
                if np.array_equal(self.agent_state,obs):
                    hit = True

        if (hit):
            reward = -1
            done = True

        elif np.array_equal(self.agent_state, self.goal_state):
            reward = 1
            done = True

        else:

            newdist = np.sum(np.abs(self.agent_state-self.goal_state))

            reward = (self.distanceFromgoal - newdist)*self.stepReward

            self.distanceFromgoal = newdist
        
        return reward, done

    def onehotrep(self):

        # onehot = np.zeros(self.rows*self.cols)
        # onehot[self.agent_state[0]*self.cols+self.agent_state[1]] = 1
        # return onehot

        return self.agent_state


if __name__=="__main__":
    
    world = GridWorld(display=True, seed = 0 , obstacles=[np.asarray([1,2])])
    for i in range(100):
        print ("here")
        state = world.reset()
        print (state)
        totalReward = 0
        done = False
        while not done:

            action = world.takeUserAction()
            next_state, reward,done,_ = world.step(action)
            #print(world.agent_state)
            #print(next_state)
            totalReward+=reward
            if done:
                break

            print("reward for the run : ", totalReward)


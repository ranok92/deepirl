import numpy as np
import torch
import time
import pdb
import sys
sys.path.insert(0, '..')
import utils  # NOQA: E402

with utils.HiddenPrints():
    import pygame

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
        obstacles = [],
        display = True,
        is_onehot = True,
        is_random = False,
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
        self.is_onehot = is_onehot
        self.is_random = is_random
        # these wrappers ensure correct output format
        self.step_wrapper = step_wrapper
        self.reset_wrapper = reset_wrapper

        if goal_state is None:
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

        '''
        this decides the state information based on whether 
        onehot information is needed or not.
        if onehot : the state becomes a onehot representation of
                    agent_state
            else  : This is the general representation of a particular
                    state for this  environment :

                        A list of numpy arrays, where 
                            index[0] : agent_state
                            index[1] : goal_state
                            index[2] : obs1
                            index[3] : obs2
                             and so on.

        If you need a different representation for your experiment,
        create new feature extractor methods in the feature extractor
        folder and use any of the above state representation from the 
        environment to create your desired feature. DO NOT create any
        specific feature extractor functions in the environment itself.

        '''         
        if self.is_onehot:
            self.state = self.onehotrep()
        else:
            self.state = {}
            self.state['agent_state'] = self.agent_state
            self.state['agent_head_dir'] = 0 #starts heading towards top
            self.state['goal_state'] = self.goal_state
            if self.obstacles is not None:
                self.state['obstacles'] = self.obstacles
            else:
                self.state['obstacles'] = self.obstacles

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

        print("environment initialized with goal state :",self.goal_state)
        self.spec = MockSpec(1.0)

        #this flag states if control has been released
        #if true, the state will not change with any action

        self.release_control = False
        self.agent_spawn_clearance = 2
        self.goal_spawn_clearance = 2


    def reset(self):

        num_obs = len(self.obstacles)

        #if this flag is true, the position of the obstacles and the goal 
        #change with each reset
        dist_g = self.goal_spawn_clearance
        if self.is_random:
            self.obstacles = []
            for i in range(num_obs):

                obs_pos = np.asarray([np.random.randint(0,self.rows),np.random.randint(0,self.cols)])
                self.obstacles.append(obs_pos)


            while True:
                flag = False
                self.goal_state = np.asarray([np.random.randint(0,self.rows),np.random.randint(0,self.cols)])

                for i in range(num_obs):
                    if np.linalg.norm(self.obstacles[i]-self.goal_state) < dist_g:

                        flag = True
                if not flag:
                    break

        dist = self.agent_spawn_clearance
        while True:
            flag = False
            self.agent_state = np.asarray([np.random.randint(0,self.rows),np.random.randint(0,self.cols)])
            for i in range(num_obs):
                if np.linalg.norm(self.obstacles[i]-self.agent_state) < dist:
                    flag = True

            if not flag:
                break


        self.distanceFromgoal = np.sum(np.abs(self.agent_state-self.goal_state))
        self.release_control = False
        if self.is_onehot:
            self.state = self.onehotrep()
        else:

            self.state = {}
            self.state['agent_state'] = self.agent_state
            self.state['agent_head_dir'] = 0 #starts heading towards top
            self.state['goal_state'] = self.goal_state

            self.state['release_control'] = self.release_control
            #if self.obstacles is not None:
            self.state['obstacles'] = self.obstacles


        if self.display:
            self.gameDisplay = pygame.display.set_mode((self.cols*self.cellWidth,self.rows*self.cellWidth))
            pygame.display.set_caption('Your friendly grid environment')
            self.render()

        if self.is_onehot:
            self.state = self.reset_wrapper(self.state)
        return self.state


    #action is a number which points to the index of the action to be taken
    def step(self,action):
        #print('printing the keypress status',self.agent_action_keyboard)
        if not self.release_control:
            self.agent_state = np.maximum(np.minimum(self.agent_state+self.actionArray[action],self.upperLimit),self.lowerLimit)
        reward, done = self.calculateReward()

        #if you are done ie hit an obstacle or the goal
        #you leave control of the agent and you are forced to
        #suffer/enjoy the consequences of your actions for the
        #rest of your miserable/awesome life

        if self.display:
            self.render()

        # step should return fourth element 'info'
        if self.is_onehot:
            self.state = self.onehotrep()
        else:
            #just update the position of the agent
            #the rest of the information remains the same

            #added new
            if not self.release_control:
                self.state['agent_state'] = self.agent_state
                if action!=4:
                    self.state['agent_head_dir'] = action 

        if self.is_onehot:

            self.state, reward, done, _ = self.step_wrapper(
                self.state,
                reward,
                done,
                None
            )
            
        if done:
            self.release_control = True

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

        onehot = np.zeros(self.rows*self.cols)
        onehot[self.agent_state[0]*self.cols+self.agent_state[1]] = 1
        return onehot

        #return self.agent_state


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


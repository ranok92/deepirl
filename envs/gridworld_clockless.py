import numpy as np
import os
import torch
import time
import pdb
import sys
from PIL import Image
from copy import copy
from collections import defaultdict
sys.path.insert(0, '..')
import utils  # NOQA: E402

with utils.HiddenPrints():
    import pygame


'''
class Obstacles:

    def __init__(self,
                 position=None, 
                 speed=0, 
                 width=10,
                 dynamic_model=None
                 ):
        if position is None:
            self.position = (0,0)
        else:
            self.position = position

        self.speed = speed
        self.width = width
        self.dynamics_model = None
'''

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
        display = False,
        is_onehot = True,
        is_random = False,
        step_reward=0.001,
        obs_width=None,
        agent_width=None,
        step_size=None,
        consider_heading=False,
        buffer_from_obs=0,
        step_wrapper=utils.identity_wrapper,
        reset_wrapper=utils.identity_wrapper,
    ):

        #environment information
        np.random.seed(seed)
        pygame.init()
        #pygame.key.set_repeat(1,200)

        #changing the obstacles and agent state with dictionaries instead of numpy arrays
        #the goal state remains the same as before
        self.default_obs_template = {'id':None, 'position':None, 'orientation':None, 'speed':None}
        self.seed = seed
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.cellWidth = width
        self.buffer_from_obs = buffer_from_obs
        if obs_width is None:
            self.obs_width=self.cellWidth
        else:
            self.obs_width=obs_width

        if agent_width is None:
            self.agent_width=self.cellWidth
        else:
            self.agent_width=agent_width

        if step_size is None:
            self.step_size=self.cellWidth
        else:
            self.step_size=step_size

        self.upper_limit_goal = np.asarray([self.rows, self.cols]) - self.cellWidth/2
        self.lower_limit_goal = self.cellWidth/2 + np.asarray([0,0])

        self.upper_limit_agent = np.asarray([self.rows, self.cols]) - self.agent_width/2
        self.lower_limit_agent = self.agent_width/2 + np.asarray([0,0])

        self.upper_limit_obstacle = np.asarray([self.rows, self.cols]) - self.obs_width/2
        self.lower_limit_obstacle = self.obs_width/2 + np.asarray([0,0])
        
        self.agent_state = copy(self.default_obs_template)
        self.agent_state['position'] = np.asarray([np.random.randint(0,self.rows-1),np.random.randint(0,self.cols-1)])
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
        self.distanceFromgoal = np.sum(np.abs(self.agent_state['position']-self.goal_state))


        self.display = display
        self.gameDisplay = None
        self.gameExit = False

        #some colors for the display
        self.white = (255,255,255)
        self.black = (0,0,0)
        self.green = (0,255,0)
        self.red = (255,0,0)

        self.freeze_obstacles = False
        self.agent_action_keyboard = [False for i in range(4)]

        self.consider_heading = consider_heading

        if self.consider_heading:

            self.rel_action_table = np.asarray([[0, 1, 2, 3],
                                                [1, 2, 3, 0],
                                                [2, 3, 0, 1],
                                                [3, 0, 1, 2]])




        self.cur_heading_dir = None
        self.heading_dir_history = None

        self.pos_history = None
        self.obstacles = []
        #does not matter if none or not.
        if isinstance(obstacles,str):

            self.read_obstacles_from_image(obstacles)
            self.freeze_obstacles = True

        if isinstance(obstacles, list):
            for i in range(len(obstacles)):
                cur_obs = copy(self.default_obs_template)
                cur_obs['id'] = i
                cur_obs['position'] = obstacles[i]
                self.obstacles.append(cur_obs)


        if isinstance(obstacles, int):
            
            num_obs = obstacles
            self.obstacles = []
            for i in range(num_obs):
                
                cur_obs = copy(self.default_obs_template)
                obs_pos = np.asarray([np.random.randint(self.lower_limit_obstacle[0],self.upper_limit_obstacle[0]),
                                     np.random.randint(self.lower_limit_obstacle[1],self.upper_limit_obstacle[1])])

                cur_obs['id'] = i
                cur_obs['position'] = obs_pos
                self.obstacles.append(cur_obs)

        #pdb.set_trace()
        '''
        this decides the state information based on whether 
        onehot information is needed or not.
        if onehot : the state becomes a onehot representation of
                    agent_state
            else  : This is the general representation of a particular
                    state for this  environment :

                        A list of numpy arrays, where calc
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


        '''          
        # 0: up, 1: right, 2: down, 3: left
        self.actionArray = [np.asarray([-1,0]),np.asarray([0,1]),np.asarray([1,0]),
                            np.asarray([0,-1]),np.asarray([0,0])]

        self.action_dict = {}

        for i in range(len(self.actionArray)):
            self.action_dict[np.array2string(self.actionArray[i])] = i
        

        self.action_space = MockActionspace(len(self.actionArray))
        '''

        # augmented action space:
        #0 : up, 1:top-right 2:right and so on ... 8: top left (in a clockwise manner
        #starting from top and doing nothing)
        

        self.actionArray = [np.asarray([-1,0]),np.asarray([-1,1]),
                            np.asarray([0,1]),np.asarray([1,1]),
                            np.asarray([1,0]),np.asarray([1,-1]),
                            np.asarray([0,-1]),np.asarray([-1,-1]), np.asarray([0,0])]

        for i in range(len(self.actionArray)):

            if np.linalg.norm(self.actionArray[i])>0:
                self.actionArray[i] = self.actionArray[i] / np.linalg.norm(self.actionArray[i])

        #print(self.actionArray)

        self.action_dict = {}

        for i in range(len(self.actionArray)):
            self.action_dict[np.array2string(self.actionArray[i])] = i

        self.action_space = MockActionspace(len(self.actionArray))

        #########################################
        
        self.step_reward = step_reward

        # TODO: Remove the below mock spec in favor of gym style spec
        # creates an environment spec containing useful info, notably reward
        # threshold at which the env is considered to be solved

        #print("environment initialized with goal state :",self.goal_state)
        self.spec = MockSpec(1.0)

        #this flag states if control has been released
        #if true, the state will not change with any actioncd ..

        self.release_control = False

        #distance to be maintained between the agent and the obstacles
        self.agent_spawn_clearance = 2+self.buffer_from_obs
        #distance to be maintained between the goal and the obstacles
        self.goal_spawn_clearance = 1+self.buffer_from_obs

        self.pos_history = []


    def if_red(self,img,top,left):

        counter = 0
        thresh = self.obs_width*self.obs_width/3
        print('top',top)
        print('left',left)
        for r in range(top,top+self.obs_width,1):
            for c in range(left,left+self.obs_width,1):
                if img[c,r][0] >180 and img[c,r][1] <20 and img[c,r][2] < 20:
                    counter+=1
                if counter >= thresh:
                    return True

        return False

    def read_obstacles_from_image(self, file_path):

        if not os.path.isfile(file_path):
            print("The existing file does not exist.")
            self.obstacles = []
        
        else:

            self.obstacles = []
            img = Image.open(file_path)
            imgval = img.load()
            self.rows = img.height
            self.cols = img.width

            self.upper_limit_goal = np.asarray([self.rows, self.cols]) - self.cellWidth/2
            self.lower_limit_goal = self.cellWidth/2 + np.asarray([0,0])

            self.upper_limit_agent = np.asarray([self.rows, self.cols]) - self.agent_width/2
            self.lower_limit_agent = self.agent_width/2 + np.asarray([0,0])

            self.upper_limit_obstacle = np.asarray([self.rows, self.cols]) - self.obs_width/2
            self.lower_limit_obstacle = self.obs_width/2 + np.asarray([0,0])
            #row_norm = self.rows*self.cellWidth/img.height
            #col_norm = self.cols*self.cellWidth/img.width

            obs_counter=0
            for r in range(0,img.height-self.obs_width-1, self.obs_width):
                for c in range(0,img.width-self.obs_width-1, self.obs_width):

                    if self.if_red(imgval,r,c):
                        cur_obs = copy(self.default_obs_template)
                        cur_obs['id'] = obs_counter
                        cur_obs['position'] = int(self.obs_width/2)+np.asarray([int(r),int(c)])
                        self.obstacles.append(cur_obs)
                        obs_counter+=1
                        #pdb.set_trace()


            '''
                        flag = False
                        new_arr = np.asarray([int(r*row_norm),int(c*col_norm)])
                        for i in range(len(self.obstacles)):
                            if np.array_equal(new_arr,self.obstacles[i]):
                                flag = True
                        if not flag:

                            self.obstacles.append(self.cellWidth * np.asarray([int(r*row_norm),int(c*col_norm)]))
            '''
        print('Total obstacles :', len(self.obstacles))
        print ("Done reading obstacles.")








    def reset(self):

        num_obs = len(self.obstacles)

        #if this flag is true, the position of the obstacles and the goal 
        #change with each reset
        dist_g = self.goal_spawn_clearance
        if self.is_random:
            if not self.freeze_obstacles:
                self.obstacles = []
                for i in range(num_obs):

                    cur_obs = copy(self.default_obs_template)
                    cur_obs['id'] = i
                    cur_obs['position'] = np.asarray([np.random.randint(self.lower_limit_obstacle[0],self.upper_limit_obstacle[0]),
                                                      np.random.randint(self.lower_limit_obstacle[1],self.upper_limit_obstacle[1])])
                    self.obstacles.append(cur_obs)


            while True:
                flag = False
                self.goal_state = np.asarray([np.random.randint(self.lower_limit_goal[0],self.upper_limit_goal[0]),
                                              np.random.randint(self.lower_limit_goal[1],self.upper_limit_goal[1])])
                
                for i in range(num_obs):
                    if np.linalg.norm(self.obstacles[i]['position']-self.goal_state) < (self.cellWidth + self.obs_width)/2 * dist_g:

                        flag = True
                if not flag:
                    break

        dist = self.agent_spawn_clearance
        while True:
            flag = False
            self.agent_state['position'] = np.asarray([np.random.randint(self.lower_limit_agent[0],self.upper_limit_agent[0]),
                                                       np.random.randint(self.lower_limit_agent[1],self.upper_limit_agent[1])])

            for i in range(num_obs):
                if np.linalg.norm(self.obstacles[i]['position']-self.agent_state['position']) < (self.agent_width + self.obs_width)/2 * dist:
                    flag = True

            if not flag:
                #print('self agent :',self.agent_state)
                break


        self.cur_heading_dir = 0
        self.heading_dir_history = []
        self.distanceFromgoal = np.linalg.norm(self.agent_state['position']-self.goal_state,1)
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

        self.pos_history.append(self.agent_state['position'])
        if self.is_onehot:
            self.state = self.reset_wrapper(self.state)
        return self.state


    #action is a number which points to the index of the action to be taken
    def step(self,action):
        #print('printing the keypress status',self.agent_action_keyboard)
        
        if not self.release_control:

            
            if self.consider_heading and action != 8:
            ##### this block is when the action space is 4 #####
                #action = self.rel_action_table[self.cur_heading_dir,action]
                #self.state['agent_head_dir'] = action 
                #self.cur_heading_dir = action
            #####################################################
            #if heading is considered and the action is not 'stay in the previous position'
            #change the action taking into account the previous heading direction
            #and also update the heading direction based on the current action               
                action = (self.cur_heading_dir + action)%8
                self.cur_heading_dir = action
                self.heading_dir_history.append(self.cur_heading_dir)

            self.agent_state['position'] = np.maximum(np.minimum(self.agent_state['position']+ \
                              (self.step_size * self.actionArray[action]),self.upper_limit_agent),self.lower_limit_agent)
        
        if not np.array_equal(self.pos_history[-1],self.agent_state['position']):
            self.pos_history.append(self.agent_state['position'])
        reward, done = self.calculate_reward()

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


    def check_overlap(self,temp_pos,obs_pos,width,buffer_val):
        #if true, that means there is an overlap

        boundary = self.agent_width/2
        distance_to_maintain = boundary+(width/2)+buffer_val
        #pdb.set_trace()
        if abs(temp_pos[0] - obs_pos[0]) < distance_to_maintain and abs(temp_pos[1] - obs_pos[1]) < distance_to_maintain:

            return True
        else:
            return False


    def calculate_reward(self):

        hit = False
        done = False


        if self.obstacles is not None:
            for obs in self.obstacles:
                if self.check_overlap(self.agent_state['position'], obs['position'], self.obs_width, self.buffer_from_obs):
                    hit = True

        if (hit):
            reward = -1
            done = True

        elif self.check_overlap(self.agent_state['position'] ,self.goal_state, self.cellWidth, 0):
            reward = 1
            done = True

        else:

            newdist = np.linalg.norm(self.agent_state['position']-self.goal_state,1)

            reward = (self.distanceFromgoal - newdist)*self.step_reward

            self.distanceFromgoal = newdist
        
        return reward, done

    def onehotrep(self):

        onehot = np.zeros(self.rows*self.cols)
        onehot[self.agent_state[0]*self.cols+self.agent_state[1]] = 1
        return onehot

        #return self.agent_state


if __name__=="__main__":
    
    world = GridWorldClockless(is_onehot=False, is_random=True, rows=100, cols=100, seed = 0 , obstacles=5)
    import time 

    start_time = time.time()
    for i in range(100):
        #print ("here")
        state = world.reset()
        #print (state)
        totalReward = 0
        done = False
        frame = 0
        pdb.set_trace()
        while frame < 1000:

            #action = world.takeUserAction()
            action = np.random.randint(4)
            next_state, reward,_,_ = world.step(action)
            #print(world.agent_state)
            #print(next_state)
            #totalReward+=reward
            frame+=1

            #print("reward for the run : ", totalReward)


    print("--- %s seconds ---" % (time.time() - start_time))
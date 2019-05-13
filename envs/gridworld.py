import numpy as np
import torch
import time
import pdb
import sys
import math
sys.path.insert(0, '..')
from featureExtractor.gridworld_featureExtractor import SocialNav,LocalGlobal,FrontBackSideSimple

from itertools import count
import utils  # NOQA: E402
from envs.gridworld_clockless import GridWorldClockless

with utils.HiddenPrints():
    import pygame

class GridWorld(GridWorldClockless):

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
        is_onehot = True,
        is_random = False,
        stepReward=0.001,
        step_wrapper=utils.identity_wrapper,
        reset_wrapper=utils.identity_wrapper,
    ):
        super().__init__(seed = seed,
                       rows = rows,
                       cols = cols,
                       width = width,
                       goal_state = goal_state,
                       obstacles = obstacles,
                       display = display,
                       is_onehot = is_onehot,
                       is_random = is_random,
                       stepReward= stepReward,
                       step_wrapper=step_wrapper,
                       reset_wrapper=reset_wrapper)
        self.clock = pygame.time.Clock()

        self.tickSpeed = 60
        if obstacles=='By hand':

            self.obstacles = self.draw_obstacles_on_board()


    def draw_obstacles_on_board(self):


        print("printing from here")
     
        self.clock.tick(self.tickSpeed)
        self.gameDisplay = pygame.display.set_mode((self.cols*self.cellWidth,self.rows*self.cellWidth))

        self.gameDisplay.fill((255,255,255))

        obstacle_list = []
        RECORD_FLAG = False
        while True:

            p = pygame.event.get()
            for event in p:

                if event.type== pygame.MOUSEBUTTONDOWN:

                    RECORD_FLAG=True
                if event.type==pygame.MOUSEBUTTONUP:

                    RECORD_FLAG = False

                if event.type==pygame.MOUSEMOTION and RECORD_FLAG:
                    #record the coordinate of the mouse
                    #convert that to a location in the gridworld
                    #store that location into the obstacle_list
                    
                    grid_loc = ([math.floor(event.pos[1]/self.cellWidth), math.floor(event.pos[0]/self.cellWidth)])
                    
                    if grid_loc not in obstacle_list:
                        obstacle_list.append(grid_loc)
                    
                
                #the recording stops when the key 'q' is pressed
                if event.type== pygame.KEYDOWN: #

                    if event.key==113:
                        
                        self.obstacle_list = obstacle_list
                        return obstacle_list

        

        return None
    

    def render(self):

        #render board
        self.clock.tick(self.tickSpeed)

        self.gameDisplay.fill(self.white)

        #render obstacles
        if self.obstacles is not None:
            for obs in self.obstacles:
                pygame.draw.rect(self.gameDisplay, self.red, [obs[1]*self.cellWidth,obs[0]*self.cellWidth,self.cellWidth, self.cellWidth])
            
        #render goal
        pygame.draw.rect(self.gameDisplay, self.green, [self.goal_state[1]*self.cellWidth, self.goal_state[0]*self.cellWidth,self.cellWidth, self.cellWidth])
        #render agent
        pygame.draw.rect(self.gameDisplay, self.black,[self.agent_state[1]*self.cellWidth, self.agent_state[0]*self.cellWidth, self.cellWidth, self.cellWidth])
        pygame.display.update()
        return 0

    #arrow keys for direction
    def take_user_action(self):
        self.clock.tick(self.tickSpeed)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                key = pygame.key.get_pressed()
                if key[pygame.K_UP]:
                    return 0,True
                if key[pygame.K_RIGHT]:
                    return 1,True
                if key[pygame.K_LEFT]:
                    return 3,True
                if key[pygame.K_DOWN]:
                    return 2,True

        return 4,False


#created this to trace the trajectory of the agents whose trajectory informations are provided in the 
#master list
    def draw_trajectories(self, trajectory_master_list):

        self.reset()
        color = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        counter = 0
        for trajectory_list in trajectory_master_list:
            for trajectory_run in trajectory_list:
                pygame.draw.lines(self.gameDisplay,color[counter],False,trajectory_run)
            counter+=1

        pygame.image.save(self.gameDisplay,'traced_trajectories')


if __name__=="__main__":

    featExt = FrontBackSideSimple(fieldList = ['agent_state','goal_state','obstacles']) 
    world = GridWorld(display=True, is_onehot = False ,seed = 0 , obstacles='By hand',rows = 50 , cols = 50 , width = 10)
    for i in range(100):
        print ("here")
        state = world.reset()
        state = featExt.extract_features(state)
        totalReward = 0
        done = False

        states = []
        states.append(state)
        for i in count(0):
            t = 0
            while t < 20:
                action,flag = world.take_user_action()
                next_state, reward,done,_ = world.step(action)
                state  = featExt.extract_features(next_state)
                if flag:
                    t+=1
                    print(state)
                    states.append(state)
                if t>20 or done:
                    break

            print("reward for the run : ", totalReward)
            print("the states in the traj :", states)
            break


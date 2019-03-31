import numpy as np
import torch
import time
import pdb
import sys
sys.path.insert(0, '..')
from featureExtractor.gridworld_featureExtractor import SocialNav


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

        self.tickSpeed = 200
 

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
    def takeUserAction(self):
        self.clock.tick(self.tickSpeed)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print("here")
                key = pygame.key.get_pressed()
                if key[pygame.K_UP]:
                    return 0
                if key[pygame.K_RIGHT]:
                    return 1
                if key[pygame.K_LEFT]:
                    return 3
                if key[pygame.K_DOWN]:
                    return 2
        return 4


if __name__=="__main__":

    featExt = SocialNav(fieldList = ['agent_state','goal_state']) 
    world = GridWorld(display=True, is_onehot = False ,seed = 0 , obstacles=[np.asarray([1,2])])
    for i in range(100):
        print ("here")
        state = world.reset()
        print (state)
        totalReward = 0
        done = False
        for i in range(300):

            action = world.takeUserAction()
            next_state, reward,done,_ = world.step(action)
            #print(world.agent_state)
            #print("next state :", next_state)
            print("Features extracted from next state :",
                featExt.extract_features(next_state))
            totalReward+=reward


            print("reward for the run : ", totalReward)


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
        show_trail = False
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
        self.show_trail = show_trail
        self.gameDisplay = pygame.display.set_mode((self.cols*self.cellWidth,self.rows*self.cellWidth))


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
        if self.show_trail:
            self.draw_trajectory()

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



    def close_game(self):

        pygame.quit()
#created this to trace the trajectory of the agents whose trajectory informations are provided in the 
#master list


    def draw_arrow(self, base_position , next_position):

        #base_position = (row,col)

        #draw the stalk
        arrow_width  = self.cellWidth*.3 #in pixels
        base_pos_pixel = (base_position+.5)*self.cellWidth
        next_pos_pixel = (next_position+.5)*self.cellWidth
       
        #draw the head
        ref_pos = base_pos_pixel+(next_pos_pixel-base_pos_pixel)*.35

        if base_position[0]==next_position[0]:
            #same row (movement left/right)
            gap = (next_pos_pixel[1]-base_pos_pixel[1])*.45
            pygame.draw.line(self.gameDisplay, (0,0,0),
                            (base_pos_pixel[1],base_pos_pixel[0]),
                            (next_pos_pixel[1]-gap,next_pos_pixel[0]))

            
            pygame.draw.polygon(self.gameDisplay,(0,0,0),
                            (
                            (ref_pos[1],ref_pos[0]+(arrow_width/2)),
                            (next_pos_pixel[1]-gap,next_pos_pixel[0]),
                            (ref_pos[1],ref_pos[0]-(arrow_width/2))  ),
                            0
                            )
            
        if base_position[1]==next_position[1]:

            gap = (next_pos_pixel[0]-base_pos_pixel[0])*.45
            pygame.draw.line(self.gameDisplay, (0,0,0),
                            (base_pos_pixel[1],base_pos_pixel[0]),
                            (next_pos_pixel[1],next_pos_pixel[0]-gap))

            pygame.draw.polygon(self.gameDisplay,(0,0,0),
                (
                (ref_pos[1]+(arrow_width/2),ref_pos[0]),
                (ref_pos[1]-(arrow_width/2),ref_pos[0]),
                (next_pos_pixel[1],next_pos_pixel[0]-gap)   ),
                0
                )


    def draw_trajectory(self):

        arrow_length = 1
        arrow_head_width = 1
        arrow_width = .1
        #denotes the start and end positions of the trajectory
        
        rad = int(self.cellWidth*.4)
        start_pos=(self.pos_history[0]+.5)*self.cellWidth
        end_pos=(self.pos_history[-1]+0.5)*self.cellWidth 

        pygame.draw.circle(self.gameDisplay,(0,255,0),
                            (int(start_pos[1]),int(start_pos[0])),
                            rad)

        pygame.draw.circle(self.gameDisplay,(0,0,255),
                    (int(end_pos[1]),int(end_pos[0])),
                    rad)

        for count in range(len(self.pos_history)-1):
            #pygame.draw.lines(self.gameDisplay,color[counter],False,trajectory_run)
            self.draw_arrow(self.pos_history[count],self.pos_history[count+1])
    

    def reset(self):

        pygame.image.save(self.gameDisplay,'traced_trajectories.png')
        self.pos_history = []

        num_obs=len(self.obstacles)

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

        self.pos_history.append(self.agent_state)

 
        pygame.display.set_caption('Your friendly grid environment')
        self.render()

        if self.is_onehot:
            self.state = self.reset_wrapper(self.state)
        return self.state


        #pygame.image.save(self.gameDisplay,'traced_trajectories')


if __name__=="__main__":

    featExt = FrontBackSideSimple(fieldList = ['agent_state','goal_state','obstacles']) 
    world = GridWorld(display=True, is_onehot = False, 
                        seed = 0 , obstacles='/home/thalassa/akonar/Pictures/test_map.jpg', show_trail=True,
                        rows = 50 , cols = 50 , width = 10)
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
                    print(world.pos_history)
                    states.append(state)
                if t>20 or done:
                    break

            print("reward for the run : ", totalReward)
            print("the states in the traj :", states)
            break


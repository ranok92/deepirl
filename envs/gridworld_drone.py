import numpy as np
import torch
import time
import pdb
import sys
import math
import os
sys.path.insert(0, '..')
from featureExtractor.gridworld_featureExtractor import SocialNav,LocalGlobal,FrontBackSideSimple

from itertools import count
import utils  # NOQA: E402
from envs.gridworld import GridWorld

with utils.HiddenPrints():
    import pygame

class GridWorldDrone(GridWorld):

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
        show_trail = False,
        annotation_file = None,
        subject = None,
        omit_annotation = None #will be used to test a policy
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
                       reset_wrapper=reset_wrapper,
                       show_trail = show_trail,
                       )


        self.gameDisplay = pygame.display.set_mode((self.cols,self.rows))
        self.annotation_file = annotation_file #the file from which the video information will be used
        self.annotation_dict = {}
        self.current_frame = 0
        self.final_frame = -1
        self.initial_frame = 999999999999 #a large number
        self.subject = subject
        self.omit_annotation = omit_annotation
        self.agent_state = None
        self.goal_state = None
        annotation_list = []

        if not os.path.isfile(annotation_file):
            print("The annotation file does not exist.")
            return 0

        with open(self.annotation_file) as f:

            for line in f:
                line = line.strip().split(' ')
                annotation_list.append(line)


        #converting the list to a dictionary, where the keys are the frame number 
        #and for each key there is a list of entries providing the annotation information
        #for that particular frame
        

        print("Loading information. . .")
        subject_final_frame = -1
        for entry in annotation_list:

            #checking for omission
            if self.omit_annotation is not None:

                if entry[0] == self.omit_annotation:
                    continue

            #checking for goal state of the given subject
            if self.subject is not None:
                if int(entry[0])==self.subject:
                    if subject_final_frame < int(entry[5]):
                        subject_final_frame = int(entry[5])
                        left = int(entry[1])
                        top = int(entry[2])
                        width = int(entry[3]) - left
                        height = int(entry[4]) - top
                        self.goal_state = np.array([top+(height/2),left+(width/2)])

            #populating the dictionary
            if entry[5] not in self.annotation_dict: #if frame is not present in the dict
                self.annotation_dict[entry[5]] = []
            
            self.annotation_dict[entry[5]].append(entry)                

            if self.subject is None:
                if self.initial_frame > int(entry[5]):
                    self.initial_frame = int(entry[5])

                if self.final_frame < int(entry[5]):
                    self.final_frame = int(entry[5])
            else:
                if int(entry[0])==self.subject:

                    if self.initial_frame > int(entry[5]):
                        self.initial_frame = int(entry[5])

                    if self.final_frame < int(entry[5]):
                        self.final_frame = int(entry[5])


      

        print('Done loading information.')
        print('initial_frame', self.initial_frame)
        print('final_frame', self.final_frame)
        print('cellWidth', self.cellWidth)
    

    def get_state_from_frame(self,frame_info):

        self.obstacles = []
        for element in frame_info:

            if element[6] != str(1): #they are visible
                if int(element[0]) != self.subject:

                    left = int(element[1])
                    top = int(element[2])
                    width = int(element[3]) - left
                    height = int(element[4]) - top
                    self.obstacles.append(np.array([int(top+(height/2)),int(left+(width/2))]))

                else:

                    left = int(element[1])
                    top = int(element[2])
                    width = int(element[3]) - left
                    height = int(element[4]) - top
                    self.agent_state = np.array([int(top+(height/2)),int(left+(width/2))])
            

    def render(self):

        #render board
        self.clock.tick(self.tickSpeed)

        self.gameDisplay.fill(self.white)
        #render obstacles
        if self.obstacles is not None:
            for obs in self.obstacles:
                pygame.draw.rect(self.gameDisplay, self.red, [obs[1],obs[0],self.cellWidth, self.cellWidth])
        #render goal
        if self.goal_state is not None:
            pygame.draw.rect(self.gameDisplay, self.green, [self.goal_state[1], self.goal_state[0],self.cellWidth, self.cellWidth])
        #render agent
        if self.agent_state is not None:
            pygame.draw.rect(self.gameDisplay, self.black,[self.agent_state[1], self.agent_state[0], self.cellWidth, self.cellWidth])
        if self.show_trail:
            self.draw_trajectory()

        pygame.display.update()
        return 0


    def step(self):
            #print('printing the keypress status',self.agent_action_keyboard)
            
        self.current_frame += 1
        #print('Info from curent frame :',self.current_frame)
        self.get_state_from_frame(self.annotation_dict[str(self.current_frame)])
        if self.subject is None:
            if not self.release_control:
                self.agent_state = np.maximum(np.minimum(self.agent_state+self.actionArray[action],self.upperLimit),self.lowerLimit)
            
            if not np.array_equal(self.pos_history[-1],self.agent_state):
                self.pos_history.append(self.agent_state)
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
            if self.subject is None:
                if not self.release_control:
                    self.state['agent_state'] = self.agent_state
                    if action!=4:
                        self.state['agent_head_dir'] = action

                    self.state['obstacles'] = self.obstacles 

        if self.is_onehot:

            self.state, reward, done, _ = self.step_wrapper(
                self.state,
                reward,
                done,
                None
            )
        
        if self.subject is None:
            if done:
                self.release_control = True

            return self.state, reward, done, None

        else:

            return self.state, 0, False, None


    def reset(self):

        #pygame.image.save(self.gameDisplay,'traced_trajectories.png')

        self.current_frame = self.initial_frame
        self.pos_history = []
        #if this flag is true, the position of the obstacles and the goal 
        #change with each reset
        dist_g = self.goal_spawn_clearance
        
        self.get_state_from_frame(self.annotation_dict[str(self.current_frame)])
        num_obs = len(self.obstacles)

        #placing the obstacles



        #only for the goal and the agent when the subject is not specified speicfically.
        '''
        if self.is_random:

            #placing the goal
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

        '''
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
        if self.display:
            self.render()

        if self.is_onehot:
            self.state = self.reset_wrapper(self.state)
        return self.state


        #pygame.image.save(self.gameDisplay,'traced_trajectories')



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
    




if __name__=="__main__":

    #featExt = FrontBackSideSimple(fieldList = ['agent_state','goal_state','obstacles']) 
    feat_ext = LocalGlobal(window_size=3, fieldList=['agent_state', 'goal_state','obstacles'])
    world = GridWorldDrone(display=True, is_onehot = False, 
                        seed = 0, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        annotation_file='/home/thalassa/akonar/Study/deepirl/envs/stanford_drone_subset/annotations/bookstore/video3/annotations.txt',
                        subject=9,                        
                        rows = 1088, cols = 1424, width = 10)
    print ("here")

    world.reset()
    while world.current_frame < world.final_frame:
        world.step()


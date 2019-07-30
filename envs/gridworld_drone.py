import numpy as np
import torch
import time
import pdb
import sys
import math
import os
sys.path.insert(0, '..')

from envs.gridworld_clockless import MockActionspace, MockSpec
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
        seed=7,
        rows=10,
        cols=10,
        width=10,
        goal_state=None,
        obstacles=None,
        display=True,
        is_onehot=True,
        is_random=False,
        stepReward=0.001,
        step_wrapper=utils.identity_wrapper,
        reset_wrapper=utils.identity_wrapper,
        show_trail=False,
        annotation_file=None,
        subject=None,
        obs_width=10,
        step_size=10,
        agent_width=10,
        show_comparison=False,
        tick_speed=30
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
                       obs_width=obs_width,
                       agent_width=agent_width,
                       step_size=step_size
                       )
        if display:

            self.gameDisplay = pygame.display.set_mode((self.cols,self.rows))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.env_font = pygame.font.SysFont('Comic Sans MS',20)
            self.tickSpeed = tick_speed
            self.show_comparison = show_comparison
            self.ghost = None
            self.ghost_state =None



        self.annotation_file = annotation_file #the file from which the video information will be used
        self.annotation_dict = {}
        self.pedestrian_dict = {}
        self.current_frame = 10
        self.final_frame = -1
        self.initial_frame = 999999999999 #a large number
        self.subject = subject
        self.max_obstacles = None
        self.agent_state = None
        self.goal_state = None
        self.agent_action_flag = False
        self.obstacle_width = obs_width
        self.step_size = step_size
        self.show_trail = show_trail
        self.annotation_list = []
        self.skip_list = [] #dont consider these pedestrians as obstacles

        self.upperLimit = np.asarray([self.rows-1, self.cols-1])
        self.lowerLimit = np.asarray([0,0])
        ############# this comes with the change in the action space##########
        self.actionArray = [np.asarray([-1,0]),np.asarray([-1,1]),
                            np.asarray([0,1]),np.asarray([1,1]),
                            np.asarray([1,0]),np.asarray([1,-1]),
                            np.asarray([0,-1]),np.asarray([-1,-1]), np.asarray([0,0])]

        self.action_dict = {}

        for i in range(len(self.actionArray)):
            self.action_dict[np.array2string(self.actionArray[i])] = i

        self.action_space = MockActionspace(len(self.actionArray))
        #self.spec = MockSpec(1.0)

        ######################################################################
        self.stepReward = stepReward
        self.generate_annotation_list()
        self.generate_pedestrian_dict()
        self.generate_annotation_dict_universal()



    def generate_annotation_list(self):
        '''
        Reads lines from an annotation file and creates a list
        '''
        if not os.path.isfile(self.annotation_file):
            print("The annotation file does not exist.")
            return 0

        with open(self.annotation_file) as f:

            for line in f:
                line = line.strip().split(' ')
                self.annotation_list.append(line)



    def generate_annotation_dict(self):

        #converting the list to a dictionary, where the keys are the frame number 
        #and for each key there is a list of entries providing the annotation information
        #for that particular frame
        #for stanford dataset format

        print("Loading information. . .")
        subject_final_frame = -1
        for entry in self.annotation_list:

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
    

    def generate_pedestrian_dict(self):
        '''
        Unlike the annotation dict, where the frames are the keys and the information is stored
        based on each frame. Here the information is stored based on the pedestrians i.e. each pedestrian
        corresponds to a key in the dictionary and the corresponding to that key is a list consisting of the 
        trajectory information of that particular pedestrian
        '''
        #the entries are of the format : frame_no, id, y_coord, x_coord
        for entry in self.annotation_list:

            if entry[1] not in self.pedestrian_dict.keys():
                self.pedestrian_dict[str(entry[1])] = []

            self.pedestrian_dict[str(entry[1])].append(entry)
        #pdb.set_trace()



    def generate_annotation_dict_universal(self):
        '''
        Reads information from files with the following (general) format
         frame , id, y_coord, x_coord
        '''
        #converting the list to a dictionary, where the keys are the frame number 
        #and for each key there is a list of entries providing the annotation information
        #for that particular frame
        

        print("Loading information. . .")
        subject_final_frame = -1
        if self.subject is not None:
            self.skip_list.append(int(self.subject))
        for entry in self.annotation_list:
            #checking for goal state of the given subject
            if self.subject is not None:
                #pdb.set_trace()
                if float(entry[1])==self.subject:
                    if subject_final_frame < int(entry[0]):
                        subject_final_frame = int(entry[0])
                        self.goal_state = np.array([float(entry[2]),float(entry[3])])

            #populating the dictionary
            if entry[0] not in self.annotation_dict: #if frame is not present in the dict
                self.annotation_dict[entry[0]] = []
            
            self.annotation_dict[entry[0]].append(entry)                

            if self.subject is None:
                if self.initial_frame > int(entry[0]):
                    self.initial_frame = int(entry[0])

                if self.final_frame < int(entry[0]):
                    self.final_frame = int(entry[0])
            else:
                if float(entry[1])==self.subject:

                    if self.initial_frame > int(entry[0]):
                        self.initial_frame = int(entry[0])

                    if self.final_frame < int(entry[0]):
                        self.final_frame = int(entry[0])
      

        print('Done loading information.')
        print('initial_frame', self.initial_frame)
        print('final_frame', self.final_frame)
        print('cellWidth', self.cellWidth)



    def get_state_from_frame_universal(self, frame_info):
        '''
        For processed datasets
        '''
        self.obstacles = []
        for element in frame_info:

            if float(element[1]) not in self.skip_list:

                self.obstacles.append(np.array([float(element[2]),float(element[3])]))

            if float(element[1]) == self.subject:

                self.agent_state = np.array([float(element[2]),float(element[3])])
                self.state['agent_state'] = self.agent_state
            if float(element[1]) == self.ghost:

                self.ghost_state = np.array([float(element[2]),float(element[3])])


        self.state['obstacles'] = self.obstacles 


    def get_state_from_frame(self,frame_info):
        '''
        For stanford dataset
        '''
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
                pygame.draw.rect(self.gameDisplay, self.red, [obs[1]-(self.obs_width/2),obs[0]-(self.obs_width/2), \
                                self.obs_width, self.obs_width])
        #render goal
        if self.goal_state is not None:
            pygame.draw.rect(self.gameDisplay, self.green, [self.goal_state[1]-(self.cellWidth/2), self.goal_state[0]- \
                             (self.cellWidth/2),self.cellWidth, self.cellWidth])
        #render agent
        if self.agent_state is not None:
            pygame.draw.rect(self.gameDisplay, self.black,[self.agent_state[1]-(self.agent_width/2), self.agent_state[0]- \
                            (self.agent_width/2), self.agent_width, self.agent_width])
        if self.ghost_state is not None:
            pygame.draw.rect(self.gameDisplay, (220,220,220),[self.ghost_state[1]-(self.agent_width/2), self.ghost_state[0]- \
                            (self.agent_width/2), self.agent_width, self.agent_width], 1)

        if self.show_trail:
            self.draw_trajectory()

        pygame.display.update()
        return 0


    def step(self, action=None):
            #print('printing the keypress status',self.agent_action_keyboard)
            
        print('Info from curent frame :',self.current_frame)

        if str(self.current_frame) in self.annotation_dict.keys():
            self.get_state_from_frame_universal(self.annotation_dict[str(self.current_frame)])

        if self.subject is None:
            if not self.release_control:

                if action is not None:
                    if isinstance(action,int):
                        self.agent_state = np.maximum(np.minimum(self.agent_state+ \
                                           self.step_size*self.actionArray[action],self.upperLimit),self.lowerLimit)
                    else:
                        self.agent_state = np.maximum(np.minimum(self.agent_state+ \
                                           self.step_size*action,self.upperLimit),self.lowerLimit)

                    #print("Agent :",self.agent_state)
                
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
                    if action!=8:
                        self.state['agent_head_dir'] = action


        if self.is_onehot:

            self.state, reward, done, _ = self.step_wrapper(
                self.state,
                reward,
                done,
                None
            )
        
        self.current_frame += 1

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
        
        self.get_state_from_frame_universal(self.annotation_dict[str(self.current_frame)])
        num_obs = len(self.obstacles)

        #placing the obstacles



        #only for the goal and the agent when the subject is not specified speicfically.
        
        if self.subject is None:
            
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

    def reset_and_replace(self):

        no_of_peds = len(self.pedestrian_dict.keys())
        cur_ped = np.random.randint(no_of_peds)

        #cur_ped = 20
        if self.show_comparison:
            self.ghost = cur_ped
        self.skip_list = []
        self.skip_list.append(cur_ped)
        self.current_frame = int(self.pedestrian_dict[str(cur_ped)][0][0]) #frame from the first entry of the list
        print('Current frame', self.current_frame)
        self.agent_state = np.asarray([float(self.pedestrian_dict[str(cur_ped)][0][2]), \
                                      float(self.pedestrian_dict[str(cur_ped)][0][3])])

        self.goal_state = np.asarray([float(self.pedestrian_dict[str(cur_ped)][-1][2]), \
                                      float(self.pedestrian_dict[str(cur_ped)][-1][3])])

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

    #taking action from user
    def take_action_from_user(self):
        #the user will click anywhere on the board and the agent will start moving
        #directly towards the point being clicked. The actions taken in the process
        #will be registered as the action performed by the expert. The agent will keep
        #moving towards the pointer as long as the left button remains pressed. Once released
        #the agent will remain in its position.
        #Using the above method, the user will have to drag the agent across the board
        #avoiding the obstacles in the process and reaching the goal.
        #if any collision is detected or the goal is not reached the 
        #trajectory will be discarded.
        (a,b,c) = pygame.mouse.get_pressed()
        
        x = 0.0001
        y = 0.0001
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.agent_action_flag = True
            if event.type == pygame.MOUSEBUTTONUP:
                self.agent_action_flag = False
        if self.agent_action_flag:  
            (x,y) = pygame.mouse.get_pos()
            #print('x :',x, 'y :',y)
            x = x - self.agent_state[1]
            y = y - self.agent_state[0]

            x = int(x/self.step_size)
            y = int(y/self.step_size)

            x = int(np.sign(x))
            y = int(np.sign(y))
            #print(x,y)
            sign_arr = np.array([y,x])
            def_arr = np.array([1,1])
            action = sign_arr*def_arr

            '''
            if np.hypot(x,y)>_max_agent_speed:
                normalizer = _max_agent_speed/(np.hypot(x,y))
            #print x,y
            else:
                normalizer = 1
            '''
            return self.action_dict[np.array2string(sign_arr*def_arr)]

        return self.action_dict[np.array2string(np.array([0,0]))]


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
                        seed=0, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        annotation_file='./expert_datasets/data_zara/annotation/processed/crowds_zara01_processed.txt',
                        subject=None,
                        tick_speed=90, 
                        obs_width=10,
                        step_size=10,
                        agent_width=30,
                        show_comparison=True,                       
                        rows=576, cols=720, width=20)
    print ("here")
    done = False
    for i in range(20):
        world.reset_and_replace()
        done = False
        while world.current_frame < world.final_frame and not done:
            _, reward , done, _ = world.step()
            print(world.agent_state)
            print (reward, done)
 
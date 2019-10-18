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
from featureExtractor.drone_feature_extractor import DroneFeatureSAM1, DroneFeatureMinimal, DroneFeatureOccup, DroneFeatureRisk, DroneFeatureRisk_v2

from envs.drone_env_utils import InformationCollector
from alternateController.potential_field_controller import PotentialFieldController as PFController
from itertools import count
import utils  # NOQA: E402
from envs.gridworld import GridWorld
import copy
with utils.HiddenPrints():
    import pygame
    import pygame.freetype


class Pedestrian():

    def __init__(self,
                 idval=None,
                 pos=None,
                 speed=None,
                 orientation=None
                 ):
        self.id = idval
        self.position = pos
        self.speed = speed 
        self.orientation = orientation


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
            step_reward=0.001,
            step_wrapper=utils.identity_wrapper,
            reset_wrapper=utils.identity_wrapper,
            show_trail=False,
            show_orientation=False,
            annotation_file=None,
            subject=None,
            obs_width=10,
            step_size=10,
            agent_width=10,
            show_comparison=False,
            tick_speed=30,
            replace_subject=False, #this option trains the agent for the exact scenarios as seen by the expert
                              #Not an ideal thing to train on. Introduced as a debugging measure.
            external_control=True,
            consider_heading=False,
            variable_speed=False
    ):
        super().__init__(seed=seed,
                         rows=rows,
                         cols=cols,
                         width=width,
                         goal_state=goal_state,
                         obstacles=obstacles,
                         display=display,
                         is_onehot=is_onehot,
                         is_random=is_random,
                         step_reward=step_reward,
                         step_wrapper=step_wrapper,
                         reset_wrapper=reset_wrapper,
                         show_trail=show_trail,
                         obs_width=obs_width,
                         consider_heading=consider_heading,
                         agent_width=agent_width,
                         step_size=step_size
                        )
        if display:

            self.gameDisplay = pygame.display.set_mode((self.cols, self.rows))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.env_font = pygame.font.SysFont('Comic Sans MS', 20)
            self.tickSpeed = tick_speed
            
        self.show_comparison = show_comparison

        self.ghost = None
        self.ghost_state = None
        self.annotation_file = annotation_file #the file from which the video information will be used
        self.annotation_dict = {}
        self.pedestrian_dict = {}
        self.current_frame = 0
        self.final_frame = -1
        self.initial_frame = 999999999999 #a large number

        self.subject = subject
        self.cur_ped = None 

        self.max_obstacles = None
        self.agent_action_flag = False
        self.obstacle_width = obs_width
        self.step_size = step_size
        self.show_trail = show_trail
        self.annotation_list = []
        self.skip_list = [] #dont consider these pedestrians as obstacles

        ############# this comes with the change in the action space##########
        self.actionArray = [np.asarray([-1, 0]), np.asarray([-1, 1]),
                            np.asarray([0, 1]), np.asarray([1, 1]),
                            np.asarray([1, 0]), np.asarray([1, -1]),
                            np.asarray([0, -1]), np.asarray([-1, -1]), np.asarray([0, 0])]

        self.speed_array = np.array([1, 0.75, 0.5, 0.25, 0])

        self.action_dict = {}
        self.prev_action = 8
        for i in range(len(self.actionArray)):
            self.action_dict[np.array2string(self.actionArray[i])] = i

        self.action_space = MockActionspace(len(self.actionArray))
        #self.spec = MockSpec(1.0)

        ######################################################################
        self.step_reward = step_reward
        if self.annotation_file is not None:

            self.generate_annotation_list()
            self.generate_pedestrian_dict()
            self.generate_annotation_dict_universal()

        else:

            print('No annotation file provided.')


        ########### debugging ##############

        self.external_control = external_control
        self.replace_subject = replace_subject

        self.show_orientation = show_orientation
        '''
        self.ped_start_pos = {} # a dictionary that will store the starting positions of all the pedestrians
        self.ped_goal_pos = {} # a dictionary that will store the goal positions of all the pedestrians

        if self.train_exact:

            for ped in self.pedestrian_dict.keys():

                self.ped_start_pos[ped] = np.asarray([float(self.pedestrian_dict[ped][0][2]), 
                                                      float(self.pedestrian_dict[ped][0][3])])

                self.ped_goal_pos[ped] = np.asarray([float(self.pedestrian_dict[ped][-1][2]),
                                                     float(self.pedestrian_dict[ped][-1][3])])
        
        #print('Printing information :')
        #print('Pedestrian dictionary :', self.pedestrian_dict)
        #print('Pedestrian starting and ending points :')
            for ped in self.pedestrian_dict.keys():

                print('Ped :', ped, 'Starting point :', self.ped_start_pos[ped], ', Ending point :', self.ped_goal_pos[ped])
            print('Pedestrian ending point :', self.ped_goal_pos)
        '''
        ################## remove this block ###################


    def generate_annotation_list(self):
        '''
        Reads lines from an annotation file and creates a list
        '''
        if self.annotation_file is not None:
            if not os.path.isfile(self.annotation_file):
                print("The annotation file does not exist.")
                exit()
                return 0

            with open(self.annotation_file) as f:

                for line in f:
                    line = line.strip().split(' ')
                    self.annotation_list.append(line)
        else:

            self.annotation_list = []



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

        ***THERE SHOULD NOT BE ANY SKIPPING OF FRAMES***

        The format of the dictionary:

            pedestrian_dict['ped_id']['frame_no']{'pos': numpy, 'orientation': numpy, 'speed': float}
        
        '''
        #the entries are of the format : frame_no, id, y_coord, x_coord


        for entry in self.annotation_list:

            if entry[1] not in self.pedestrian_dict.keys(): #adding a new pedestrian

                self.pedestrian_dict[str(entry[1])] = {}
                self.pedestrian_dict[str(entry[1])]['initial_frame'] = str(entry[0])
                self.pedestrian_dict[str(entry[1])]['final_frame'] = str(entry[0])
                speed = None
                orientation = None
                pos = np.asarray([float(entry[2]), float(entry[3])]) #[row, col]
            else:
                pos = np.asarray([float(entry[2]), float(entry[3])]) #[row, col]
                orientation = pos - self.pedestrian_dict[str(entry[1])][str(int(entry[0])-1)]['position'] 
                speed = np.linalg.norm(orientation)

            self.pedestrian_dict[str(entry[1])][str(entry[0])] = {} #initialize the dictionary for the frame regardless of the first or any other frames

            #update the final frame everytime you encounter something bigger
            if int(self.pedestrian_dict[str(entry[1])]['final_frame']) < int(entry[0]):
                self.pedestrian_dict[str(entry[1])]['final_frame'] = str(entry[0])

            #populate the dictionary 
            '''
            the format of the dictionary : ped_dict['ped_id']['frame_id']['pos', 'orientation', 'speed']
            '''
            self.pedestrian_dict[str(entry[1])][str(entry[0])]['position'] = pos
            self.pedestrian_dict[str(entry[1])][str(entry[0])]['orientation'] = orientation
            self.pedestrian_dict[str(entry[1])][str(entry[0])]['speed'] = speed
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
        #if self.cur_ped is not None:
        #    self.skip_list.append(int(self.cur_ped))
        for entry in self.annotation_list:
            #checking for goal state of the given subject
            if self.cur_ped is not None:
                #pdb.set_trace()
                if float(entry[1])==self.subject:
                    if subject_final_frame < int(entry[0]):
                        subject_final_frame = int(entry[0])
                        self.goal_state = np.array([float(entry[2]),float(entry[3])])

            #populating the dictionary
            if entry[0] not in self.annotation_dict: #if frame is not present in the dict
                self.annotation_dict[entry[0]] = []
            
            self.annotation_dict[entry[0]].append(entry)                

            if self.cur_ped is None:
                if self.initial_frame > int(entry[0]):
                    self.initial_frame = int(entry[0])

                if self.final_frame < int(entry[0]):
                    self.final_frame = int(entry[0])
            else:
                if float(entry[1])==self.cur_ped:

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

            #populating the obstacles
            if float(element[1]) not in self.skip_list:

                obs = self.pedestrian_dict[element[1]][str(self.current_frame)]
                obs['id'] = element[1]
                self.obstacles.append(obs)
                #print(obs)
                #pdb.set_trace()
            #populating the agent
            #dont update the agent if training is going on
            if not self.external_control:
                #pdb.set_trace()
                
                if float(element[1]) == self.cur_ped:
                    agent = self.pedestrian_dict[element[1]][str(self.current_frame)]
                    self.agent_state = agent
                    self.state['agent_state'] = copy.deepcopy(self.agent_state)

            #populating the ghost
            if float(element[1]) == self.ghost:


                self.ghost_state = self.pedestrian_dict[element[1]][str(self.current_frame)]['position']


        self.state['obstacles'] = self.obstacles 


    def get_state_from_frame(self,frame_info):
        '''
        For stanford dataset
        '''
        self.obstacles = []
        for element in frame_info:

            if element[6] != str(1): #they are visible
                if int(element[0]) != self.cur_ped:

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
        font = pygame.freetype.Font(None, 15)
        self.gameDisplay.fill(self.white, [0,0, self.cols, self.rows])
        #render obstacles
        if self.obstacles is not None:
            for obs in self.obstacles:
                pygame.draw.rect(self.gameDisplay, self.red, [obs['position'][1]-(self.obs_width/2),obs['position'][0]-(self.obs_width/2), \
                                self.obs_width, self.obs_width])
                font.render_to(self.gameDisplay, 
                              (obs['position'][1]-(self.obs_width/2)-5,obs['position'][0]-(self.obs_width/2)), 
                              obs['id'], fgcolor=(0,0,0))
                if self.show_orientation:
                    if obs['orientation'] is not None: 
                        pygame.draw.line(self.gameDisplay, self.black, [obs['position'][1],obs['position'][0]], 
                                         [obs['position'][1]+obs['orientation'][1]*10, obs['position'][0]+obs['orientation'][0]*10], 2)
        #render goal
        if self.goal_state is not None:
            pygame.draw.rect(self.gameDisplay, self.green, [self.goal_state[1]-(self.cellWidth/2), self.goal_state[0]- \
                             (self.cellWidth/2),self.cellWidth, self.cellWidth])
        #render agent
        if self.agent_state is not None:
            pygame.draw.rect(self.gameDisplay, self.black,[self.agent_state['position'][1]-(self.agent_width/2), self.agent_state['position'][0]- \
                            (self.agent_width/2), self.agent_width, self.agent_width])

            if self.show_orientation:
                if self.agent_state['orientation'] is not None: 
                        pygame.draw.line(self.gameDisplay, self.black, [self.agent_state['position'][1],self.agent_state['position'][0]], 
                                         [self.agent_state['position'][1]+self.agent_state['orientation'][1]*10, self.agent_state['position'][0]+self.agent_state['orientation'][0]*10], 2)

        if self.ghost_state is not None:
            pygame.draw.rect(self.gameDisplay, (220,220,220),[self.ghost_state[1]-(self.agent_width/2), self.ghost_state[0]- \
                            (self.agent_width/2), self.agent_width, self.agent_width], 1)

        if self.show_trail:
            self.draw_trajectory()


        pygame.display.update()
        return 0


    def step(self, action=None):
        '''
        if external_control: t
            the state of the agent is updated based on the current action. 
        else:
            the state of the agent is updated based on the information from the frames

        the rest of the actions, like calculating reward and checking if the episode is done remains as usual.
        '''
        self.current_frame += 1

        if str(self.current_frame) in self.annotation_dict.keys():
            self.get_state_from_frame_universal(self.annotation_dict[str(self.current_frame)])

        if self.external_control:

            if not self.release_control:

                if action is not None:
                    if isinstance(action, int):

                        if action != 8 and self.consider_heading:
                            action = (self.cur_heading_dir + action)%8
                            self.cur_heading_dir = action
                        #self.heading_dir_history.append(self.cur_heading_dir)
                        #self.cur_heading_dir = action
                        prev_position = self.agent_state['position']
                        self.agent_state['position'] = np.maximum(np.minimum(self.agent_state['position']+ \
                                           self.step_size*self.actionArray[action],self.upper_limit_agent),self.lower_limit_agent)

                        self.agent_state['orientation'] = self.agent_state['position'] - prev_position
                        self.agent_state['speed'] = np.linalg.norm(self.agent_state['orientation'])

                        
                    else:
                        #if the action is a torch
                        if len(action.shape)==1 and a.shape[0]==1: #check if it the tensor has a single value
                            if isinstance(action.item(), int):
                                prev_position = self.agent_state['position']

                                if action!=8 and self.consider_heading:
                                    action = (self.cur_heading_dir + action)%8
                                    self.cur_heading_dir =  action
                                #self.heading_dir_history.append(self.cur_heading_dir)
                                #self.cur_heading_dir = action
                                
                                self.agent_state['position'] = np.maximum(np.minimum(self.agent_state['position']+ \
                                                               self.step_size*self.actionArray[action],
                                                               self.upper_limit_agent),self.lower_limit_agent)

                                self.agent_state['orientation'] = self.agent_state['position'] - prev_position
                                self.agent_state['speed'] = np.linalg.norm(self.agent_state['orientation'])
                    #print("Agent :",self.agent_state)
            
            #if not np.array_equal(self.pos_history[-1],self.agent_state):
            self.heading_dir_history.append(self.cur_heading_dir)

            self.pos_history.append(copy.deepcopy(self.agent_state))


        #calculate the reward and completion condition
        reward, done = self.calculate_reward(action)
        self.prev_action = action
        
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
                self.state['agent_state'] = copy.deepcopy(self.agent_state)
                if action!=8:
                    self.state['agent_head_dir'] = action


        if self.is_onehot:

            self.state, reward, done, _ = self.step_wrapper(
                self.state,
                reward,
                done,
                None
            )
        


        if self.external_control:
            if done:
                self.release_control = True

        return self.state, reward, done, None



    def calculate_reward(self, cur_action):

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

        if cur_action is not None:

            energy_spent = -np.sum(np.square(self.actionArray[cur_action]-self.actionArray[self.prev_action]))
            
            reward += energy_spent*self.step_reward*1

        #pdb.set_trace()
        return reward, done


    def reset(self):
        '''
        Resets the environment, starting the obstacles from the start.
        If subject is specified, then the initial frame and final frame is set
        to the time frame when the subject is in the scene.

        If no subject is specified then the initial frame is set to the overall 
        initial frame and goes till the last frame available in the annotation file.

        Also, the agent and goal positions are initialized at random.

        Pro tip: Use this function while training the agent. 
        '''
        #pygame.image.save(self.gameDisplay,'traced_trajectories.png')
        #########for debugging purposes###########
        if self.replace_subject:

            return self.reset_and_replace()

        else:
            
            #self.skip_list = [i for i in range(len(self.pedestrian_dict.keys()))]

            self.current_frame = self.initial_frame
            self.pos_history = []
            #if this flag is true, the position of the obstacles and the goal 
            #change with each reset
            dist_g = self.goal_spawn_clearance

            if self.annotation_file:
                self.get_state_from_frame_universal(self.annotation_dict[str(self.current_frame)])
            
            num_obs = len(self.obstacles)

            #placing the obstacles



            #only for the goal and the agent when the subject is not specified speicfically.
            
            if self.cur_ped is None:
                
                #placing the goal
                while True:
                    flag = False
                    self.goal_state = np.asarray([np.random.randint(self.lower_limit_goal[0],self.upper_limit_goal[0]),
                                                  np.random.randint(self.lower_limit_goal[1],self.upper_limit_goal[1])])

                    for i in range(num_obs):
                        if np.linalg.norm(self.obstacles[i]['position']-self.goal_state) < dist_g:

                            flag = True
                    if not flag:
                        break

                #placing the agent
                dist = self.agent_spawn_clearance
                while True:
                    flag = False
                    #pdb.set_trace()
                    self.agent_state['position'] = np.asarray([np.random.randint(self.lower_limit_agent[0],self.upper_limit_agent[0]),
                                                   np.random.randint(self.lower_limit_agent[1],self.upper_limit_agent[1])])

                    for i in range(num_obs):
                        if np.linalg.norm(self.obstacles[i]['position']-self.agent_state['position']) < dist:
                            flag = True

                    if not flag:
                        break

            
            self.release_control = False
            if self.is_onehot:
                self.state = self.onehotrep()
            else:

                self.state = {}
                self.state['agent_state'] = copy.deepcopy(self.agent_state)
                self.state['agent_head_dir'] = 0 #starts heading towards top
                self.state['goal_state'] = self.goal_state

                self.state['release_control'] = self.release_control
                #if self.obstacles is not None:
                self.state['obstacles'] = self.obstacles

            self.pos_history.append(copy.deepcopy(self.agent_state))

            self.distanceFromgoal = np.linalg.norm(self.agent_state['position']-self.goal_state,1)
            self.cur_heading_dir = 0
            self.heading_dir_history = []
            self.heading_dir_history.append(self.cur_heading_dir)

            pygame.display.set_caption('Your friendly grid environment')
            if self.display:
                self.render()

            if self.is_onehot:
                self.state = self.reset_wrapper(self.state)
            return self.state


        #pygame.image.save(self.gameDisplay,'traced_trajectories')

    def reset_and_replace(self):
        '''
        Resets the environment and replaces one of the existing pedestrians
        from the video feed in the environment with the agent. 
        Pro tip: Use this for testing the result.
        '''
        no_of_peds = len(self.pedestrian_dict.keys())
        if self.subject is None:
            while True:
                
                self.cur_ped = np.random.randint(1,no_of_peds+1)
                if str(self.cur_ped) in self.pedestrian_dict.keys():
                    break
                else:
                    print('Selected pedestrian not available.')
        else:
            self.cur_ped = self.subject

        #print('Replacing agent :', self.cur_ped)
        if self.display:
            if self.show_comparison:
                self.ghost = self.cur_ped

        self.skip_list = [] 
        self.skip_list.append(self.cur_ped)
        self.current_frame = int(self.pedestrian_dict[str(self.cur_ped)]['initial_frame']) #frame from the first entry of the list
        self.get_state_from_frame_universal(self.annotation_dict[str(self.current_frame)])

        self.agent_state = copy.deepcopy(self.pedestrian_dict[str(self.cur_ped)][str(self.current_frame)])
        #self.agent_state = np.asarray([float(self.pedestrian_dict[str(cur_ped)][0][2]), \
        #                              float(self.pedestrian_dict[str(cur_ped)][0][3])])

        self.final_frame = int(self.pedestrian_dict[str(self.cur_ped)]['final_frame'])
        print('Cur_ped : {} final frame {}'.format(self.cur_ped, self.final_frame))
        #self.goal_state = np.asarray([float(self.pedestrian_dict[str(cur_ped)][-1][2]), \
        #                              float(self.pedestrian_dict[str(cur_ped)][-1][3])])

        self.goal_state = self.pedestrian_dict[str(self.cur_ped)][str(self.final_frame)]['position']

        self.release_control = False


        if self.is_onehot:

            self.state = self.onehotrep()
        else:
            self.state = {}
            self.state['agent_state'] = copy.deepcopy(self.agent_state)
            self.state['agent_head_dir'] = 0 #starts heading towards top
            self.state['goal_state'] = self.goal_state

            self.state['release_control'] = self.release_control
            #if self.obstacles is not None:
            self.state['obstacles'] = self.obstacles

        self.pos_history = []
        self.pos_history.append(copy.deepcopy(self.agent_state))

        self.distanceFromgoal = np.linalg.norm(self.agent_state['position']-self.goal_state,1)
        self.cur_heading_dir = 0
        self.heading_dir_history = []
        self.heading_dir_history.append(self.cur_heading_dir)

        if self.display:
            pygame.display.set_caption('Your friendly grid environment')
            self.render()

        if self.is_onehot:
            self.state = self.reset_wrapper(self.state)

        #pdb.set_trace()
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
            x = x - self.agent_state['position'][1]
            y = y - self.agent_state['position'][0]

            x = int(x/self.step_size)
            y = int(y/self.step_size)

            x = int(np.sign(x))
            y = int(np.sign(y))
            #print(x,y)
            sign_arr = np.array([y,x])
            def_arr = np.array([1,1])
            action = sign_arr*def_arr

            #action = (self.prev_action + action)%8
            '''
            if np.hypot(x,y)>_max_agent_speed:
                normalizer = _max_agent_speed/(np.hypot(x,y))
            #print x,y
            else:
                normalizer = 1
            '''
            action = self.action_dict[np.array2string(sign_arr*def_arr)]
            #pdb.set_trace()
            #print( 'absolute action ',action)
            action = (action - self.cur_heading_dir)%8
            #print( ' relative action :',action)
            #self.prev_action = action
            #print('relative action :', action)
            return action

        return self.action_dict[np.array2string(np.array([0,0]))]


    def close_game(self):

        pygame.quit()
#created this to trace the trajectory of the agents whose trajectory informations are provided in the 
#master list
    

    def rollback(self, frames):
        ''' 
        Added this function primarily for reward analysis purpose.
        Provided the frames, this function rolls the environment back in time by the number of 
        frames provided
        '''
        self.current_frame = self.current_frame - frames 

        if str(self.current_frame) in self.annotation_dict.keys():
            self.get_state_from_frame_universal(self.annotation_dict[str(self.current_frame)])

        if self.external_control:

            self.agent_state = copy.deepcopy(self.pos_history[-frames-1])
            self.cur_heading_dir = self.heading_dir_history[-frames-1]

            if frames > len(self.heading_dir_history):
                print('Trying to rollback more than the size of current history!')
            else:
                for i in range(1,frames+1):

                    self.heading_dir_history.pop(-1)
                    self.pos_history.pop(-1)

        if self.release_control:
            self.release_control = False
        self.state['agent_state'] = copy.deepcopy(self.agent_state)
        if self.display:
            self.render()

        return self.state




    def draw_arrow(self, base_position , next_position):

        #base_position = (row,col)

        #draw the stalk
        arrow_width  = self.cellWidth*.3 #in pixels
        base_pos_pixel = (base_position+.5)
        next_pos_pixel = (next_position+.5)
        pdb.set_trace()

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

    '''
    feat_ext = FrontBackSideSimple(agent_width=10,
                                   obs_width=10,
                                   step_size=10,
                                   grid_size=10) 
    
    feat_drone = DroneFeatureSAM1(step_size=50)
    window_size = 15
    feat_drone_2 = DroneFeatureOccup(step_size=10, window_size=window_size)
    '''
    
    info_collector = InformationCollector(thresh=60,
                                          run_info='Potential field controller',
                                          disp_info=True)

    feat_drone = DroneFeatureSAM1(step_size=2,
                                  thresh1=10,
                                  thresh2=20)
    
    feat_drone = DroneFeatureRisk_v2(step_size=2,
                                  thresh1=15,
                                  thresh2=30)

    world = GridWorldDrone(display=True, is_onehot = False, 
                        seed=0, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        annotation_file='../envs/expert_datasets/university_students/annotation/processed/frame_skip_1/students003_processed_corrected.txt',
                        subject=7,
                        tick_speed=60, 
                        obs_width=7,
                        step_size=2,
                        agent_width=7,
                        step_reward=0.01,
                        show_comparison=False,
                        show_orientation=True,
                        external_control=False,
                        replace_subject=True, 
                        consider_heading=False,                      
                        rows=576, cols=720, width=20)

    pf_agent = PFController()
    '''
    feat_ext = LocalGlobal(window_size=9, 
                           grid_size = 10,
                           step_size = 15,
                           agent_width=10,
                           overlay_bins=True,
                           pygame_surface=world.gameDisplay,
                           obs_width=10)
'''
    print ("here")
    
    done = False
    for i in range(50):


        state = world.reset()
        feat_drone.reset()

        print(world.cur_ped)
        info_collector.reset_info(state)
        done = False
        init_frame = world.current_frame
        fin_frame = world.final_frame
        t = 1
        while world.current_frame < fin_frame:
            #action = input()

            action = world.take_action_from_user()

            #action = pf_agent.select_action(state)
            
            state, reward , done, _ = world.step(action)
            #print(reward, done)
            info_collector.collect_information_per_frame(state)
            #print(state['agent_state']['orientation'])
            #feat_drone.overlay_bins(state)
            #print(state)
            #pdb.set_trace()
            feat = feat_drone.extract_features(state)
            if t%100==0:
                world.rollback(10)
                feat_drone.rollback(10, state)
                t=1
            #feat2 = feat_drone_2.extract_features(state)
            #orientation = feat_drone.extract_features(state)
            '''
            print('Global info:')
            print(feat[0:9].reshape(3, 3))
            print('global info_goal:')
            print(feat[9:18].reshape(3, 3))

            print('Occupancy grid info:')
            print(feat[22:].reshape(window_size, window_size))
            
            pdb.set_trace()
            '''
            #print(world.agent_state)
            #print (reward, done)
            t+=1
        info_collector.collab_end_traj_results()

    info_collector.collab_end_results()
    
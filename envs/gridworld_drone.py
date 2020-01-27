import pdb
import sys
import math
import os
sys.path.insert(0, '..')
import numpy as np
from gym.spaces import Discrete, Box
from featureExtractor.gridworld_featureExtractor import SocialNav,LocalGlobal,FrontBackSideSimple
from featureExtractor.drone_feature_extractor import DroneFeatureSAM1, DroneFeatureMinimal, DroneFeatureOccup, DroneFeatureRisk, DroneFeatureRisk_v2
from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speed
from envs.drone_env_utils import InformationCollector
from alternateController.potential_field_controller import PotentialFieldController as PFController
from alternateController.social_forces_controller import SocialForcesController
from itertools import count
import utils  # NOQA: E402
from envs.gridworld import GridWorld
from envs.drone_env_utils import angle_between
import copy
with utils.HiddenPrints():
    import pygame
    import pygame.freetype


def deg_to_rad(deg):
    return deg*np.pi/180


def rad_to_deg(rad):
    return rad*180/np.pi

def get_rot_matrix(theta):
    '''
    returns the rotation matrix given a theta value(radians)
    '''
    return np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

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
            rows=100,
            cols=100,
            width=10,
            goal_state=None,
            obstacles=None,
            display=True,
            is_onehot=False,
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
            replace_subject=False, #this option trains the agent for the exact scenarios as seen 
            #by the expert Not an ideal thing to train on. Introduced as a debugging measure.
            segment_size=None,
            external_control=True,
            consider_heading=False,
            continuous_action=False
    ):
        super().__init__(seed=seed,
                         rows=rows,
                         cols=cols,
                         width=width,
                         goal_state=goal_state,
                         obstacles=obstacles,
                         display=display,
                         is_onehot=False,
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
            self.enable_rendering(tick_speed)


        self.show_comparison = show_comparison

        self.ghost = None
        self.ghost_state = None
        self.ghost_state_history = []
        self.ghost_color = (140, 0, 200)


        self.annotation_file = annotation_file#file from which the video information will be used
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

        self.continuous_action = continuous_action
        if not self.continuous_action:
            divisions = 7
            self.orient_quantization = 10
            min_val = -self.orient_quantization*(divisions-1)/2
            self.orientation_array = [min_val+(i*self.orient_quantization) for i in range(divisions)]
            divisions = 5
            self.speed_quantization = .2
            min_val = -self.speed_quantization*(divisions-1)/2
            self.max_speed = 2
            self.speed_array = [min_val+(i*self.speed_quantization) for i in range(divisions)]
            self.action_space = Discrete(len(self.orientation_array)*len(self.speed_array))
        else:
            self.max_speed = 2
            self.max_orient_change = 30
            self.action_space = Box(np.array([-.5, -self.max_orient_change]),
                                    np.array([.5, self.max_orient_change]))
            #The action array is a 2 dimensional array
            #        [change in speed, change in orientation]
        '''
        Some things to note:
            1. The orientation of the agent will be a 2d vector pointing in the direction in which the
               agent is currently heading.
            2. "cur_heading_dir" will contain the degree (integer) in which the agent is heading.
            3. Speed can only be positive and bound within a range
        '''
        #################################
        '''
        part used for taking action from user. Needs to be modified
        self.action_dict = {}
        self.prev_action = 8
        for i in range(len(self.actionArray)):
            self.action_dict[np.array2string(self.actionArray[i])] = i
        '''
        ##################################
        self.step_reward = step_reward


        if self.annotation_file is not None:

            if not os.path.isfile(self.annotation_file):
                print("The annotation file does not exist.")
                print("Starting environment without annotation file.")
                self.annotation_file = None
            else:
                self.generate_annotation_list()
                self.generate_pedestrian_dict()
                self.generate_annotation_dict_universal()

        else:

            print('No annotation file provided.')

        self.external_control = external_control
        self.replace_subject = replace_subject
        self.segment_size = segment_size
        self.show_orientation = show_orientation



    def enable_rendering(self, tick_speed):
        pygame.quit()
        pygame.init()
        self.display=True
        self.gameDisplay = pygame.display.set_mode((self.cols, self.rows))
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.env_font = pygame.font.SysFont('Comic Sans MS', 20)
        self.tickSpeed = tick_speed

    def disable_rendering(self):
        self.display=False

    def generate_annotation_list(self):
        '''
        Reads lines from an annotation file and creates a list
        '''
        if self.annotation_file is not None:
            if not os.path.isfile(self.annotation_file):
                print("The annotation file does not exist.")
                exit()

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
            #populating the agent
            #dont update the agent if training is going on
            if not self.external_control:
                if float(element[1]) == self.cur_ped:
                    agent = self.pedestrian_dict[element[1]][str(self.current_frame)]
                    self.agent_state = agent
                    self.state['agent_state'] = copy.deepcopy(self.agent_state)

            #populating the ghost
            if float(element[1]) == self.ghost:


                self.ghost_state = self.pedestrian_dict[element[1]][str(self.current_frame)]
                self.ghost_state_history.append(self.ghost_state)

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
                mag = 10 + 10 * self.agent_state['speed']

                if self.agent_state['orientation'] is not None: 
                        pygame.draw.line(self.gameDisplay, self.black, [self.agent_state['position'][1],self.agent_state['position'][0]], 
                                         [self.agent_state['position'][1]+self.agent_state['orientation'][1]*mag, self.agent_state['position'][0]+self.agent_state['orientation'][0]*mag], 
                                         2)
        if self.ghost_state is not None:

            pygame.draw.rect(self.gameDisplay, self.ghost_color, [self.ghost_state['position'][1]-(self.agent_width/2), self.ghost_state['position'][0]- \
                            (self.agent_width/2), self.agent_width, self.agent_width])

        if self.show_trail:

            self.draw_trajectory(self.pos_history, self.black)

            if self.ghost:
                self.draw_trajectory(self.ghost_state_history, self.ghost_color)


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

                    if not self.continuous_action:
                        action_orient = int(action%len(self.orientation_array))
                        action_speed = int(action/len(self.orientation_array))
                        #print(action_speed, "   ", action)
                        orient_change = self.orientation_array[action_orient]
                        speed_change = self.speed_array[action_speed]
                    else:
                        speed_change = action[0]
                        orient_change = action[1]

                    #if self.consider_heading:
                        #after 360, it comes back to 0

                    self.cur_heading_dir = (self.cur_heading_dir+orient_change)%360
                    agent_cur_speed = max(0,min(self.agent_state['speed'] + speed_change, self.max_speed))

                    prev_position = self.agent_state['position']
                    rot_mat = get_rot_matrix(deg_to_rad(-self.cur_heading_dir))
                    cur_displacement = np.matmul(rot_mat, np.array([-agent_cur_speed, 0]))
                    '''
                    cur_displacement is a 2 dim vector where the displacement is in the form:
                        [row, col]
                    '''
                    self.agent_state['position'] = np.maximum(np.minimum(self.agent_state['position']+ \
                                       cur_displacement,self.upper_limit_agent),self.lower_limit_agent)

                    self.agent_state['speed'] = agent_cur_speed
                    self.agent_state['orientation'] = np.matmul(rot_mat, np.array([-1,0]))

            self.heading_dir_history.append(self.cur_heading_dir)

            self.pos_history.append(copy.deepcopy(self.agent_state))

            if self.ghost:
                self.ghost_state_history.append(copy.deepcopy(self.ghost_state))

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

        #just update the position of the agent
        #the rest of the information remains the same

        #added new
        if not self.release_control:
            self.state['agent_state'] = copy.deepcopy(self.agent_state)
            self.state['agent_head_dir'] = self.cur_heading_dir

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
            energy_spent = 0
            reward += energy_spent*self.step_reward*1

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
        if self.replace_subject:
            return self.reset_and_replace()

        else:
            self.current_frame = self.initial_frame
            self.pos_history = []
            self.ghost_state_history = []
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
                #add speed and orientation to the agent state after it is placed successfully

                self.agent_state['speed'] = 0 #dead stop
                self.cur_heading_dir = 0 #pointing upwards
                self.agent_state['orientation'] = np.matmul(get_rot_matrix(deg_to_rad(self.cur_heading_dir)),
                                                                            np.array([self.agent_state['speed'], 0]))

            self.release_control = False


            self.state = {}
            self.state['agent_state'] = copy.deepcopy(self.agent_state)
            self.state['agent_head_dir'] = self.cur_heading_dir #starts heading towards top
            self.state['goal_state'] = self.goal_state

            self.state['release_control'] = self.release_control
            #if self.obstacles is not None:
            self.state['obstacles'] = self.obstacles

            self.pos_history.append(copy.deepcopy(self.agent_state))
            if self.ghost:
                self.ghost_state_history.append(copy.deepcopy(self.ghost_state))

            if self.ghost:
                self.ghost_state_history.append(copy.deepcopy(self.ghost_state))

            self.distanceFromgoal = np.linalg.norm(self.agent_state['position']-self.goal_state,1)
            self.cur_heading_dir = 0
            self.heading_dir_history = []
            self.heading_dir_history.append(self.cur_heading_dir)

            pygame.display.set_caption('Your not so friendly continuous environment')
            if self.display:
                self.render()

            return self.state


        #pygame.image.save(self.gameDisplay,'traced_trajectories')

    def reset_and_replace(self, ped=None):
        '''
        Resets the environment and replaces one of the existing pedestrians
        from the video feed in the environment with the agent.
        Pro tip: Use this for testing the result.
        '''
        #pdb.set_trace()
        no_of_peds = len(self.pedestrian_dict.keys())

        if self.subject is None:
            while True:
                if ped is not None:
                    self.cur_ped=ped
                    break
                else:
                    if self.is_random:
                        self.cur_ped = np.random.randint(1,no_of_peds+1)
                    else:
                        if self.cur_ped is None:
                            self.cur_ped = 1
                        else:
                            self.cur_ped += 1
                    if str(self.cur_ped) in self.pedestrian_dict.keys():
                        break
                    else:
                        print('Selected pedestrian not available.')
        else:
            self.cur_ped = self.subject

        #print('Replacing agent :', self.cur_ped)
        #if self.display:
        if self.show_comparison:
            self.ghost = self.cur_ped

        self.skip_list = [] 
        self.skip_list.append(self.cur_ped)
        if self.segment_size is None:
            
            self.current_frame = int(self.pedestrian_dict[str(self.cur_ped)]['initial_frame']) #frame from the first entry of the list
            self.final_frame = int(self.pedestrian_dict[str(self.cur_ped)]['final_frame'])
            self.goal_state = self.pedestrian_dict[str(self.cur_ped)][str(self.final_frame)]['position']
           
        else:
            first_frame = int(self.pedestrian_dict[str(self.cur_ped)]['initial_frame'])
            final_frame = int(self.pedestrian_dict[str(self.cur_ped)]['final_frame'])
            total_frames = final_frame - first_frame

            total_segments = int(total_frames/self.segment_size) + 1
            cur_segment = np.random.randint(total_segments)
            self.current_frame = first_frame + cur_segment*self.segment_size
            self.final_frame = min(self.current_frame+self.segment_size, final_frame)
            self.goal_state = self.pedestrian_dict[str(self.cur_ped)][str(self.final_frame)]['position']


        self.get_state_from_frame_universal(self.annotation_dict[str(self.current_frame)])

        self.agent_state = copy.deepcopy(self.pedestrian_dict[str(self.cur_ped)][str(self.current_frame)])
        #the starting state for any pedestrian in the dict has none for orientation and speed
        self.agent_state['speed'] = 0  #zero speed
        self.cur_heading_dir = 0
        self.agent_state['orientation'] = np.matmul(get_rot_matrix(deg_to_rad(self.cur_heading_dir)),
                                                                            np.array([self.agent_state['speed'], 0]))

        self.release_control = False


        self.state = {}
        self.state['agent_state'] = copy.deepcopy(self.agent_state)
        self.state['agent_head_dir'] = self.cur_heading_dir #starts heading towards top
        self.state['goal_state'] = self.goal_state

        self.state['release_control'] = self.release_control
        #if self.obstacles is not None:
        self.state['obstacles'] = self.obstacles

        self.pos_history = []
        self.pos_history.append(copy.deepcopy(self.agent_state))
        if self.ghost:
            self.ghost_state_history = []
            self.ghost_state = copy.deepcopy(self.agent_state)
            self.ghost_state_history.append(copy.deepcopy(self.ghost_state))

        if self.ghost:
            self.ghost_state_history = []
            self.ghost_state = copy.deepcopy(self.agent_state)
            self.ghost_state_history.append(copy.deepcopy(self.ghost_state))

        self.distanceFromgoal = np.linalg.norm(self.agent_state['position']-self.goal_state,1)
        self.heading_dir_history = []
        self.heading_dir_history.append(self.cur_heading_dir)

        if self.display:
            pygame.display.set_caption('Your not so friendly continuous environment')
            self.render()


        #pdb.set_trace()
        return self.state


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
            #pdb.set_trace()
            new_dir_vector_abs = np.asarray([y, x])
            rot_mat = get_rot_matrix(deg_to_rad(self.cur_heading_dir))
            rotated_vector = np.matmul(rot_mat, new_dir_vector_abs)
            rotated_agent_heading_vector = np.matmul(rot_mat, self.agent_state['orientation'])

            action_orient = int(len(self.orientation_array)/2)
            action_speed = int(len(self.speed_array)/2)
            #print(self.agent_state)
            mag_angle = rad_to_deg(angle_between(rotated_agent_heading_vector, rotated_vector))
            #print("The mag_angle :", mag_angle)
            orient_action = min( (len(self.orientation_array)-1)/2, mag_angle/self.orient_quantization)

            if (rotated_vector[1] < rotated_agent_heading_vector[1]):
                #the new action wants the agent to move to its relative left
                #print("moving to left")
                orient_action = (len(self.orientation_array)-1)/2 - orient_action
            else:
                #else to the relative right
                #print("moving to right")
                orient_action = (len(self.orientation_array)-1)/2 + orient_action

            target_vel = np.linalg.norm(rotated_agent_heading_vector)
            change = target_vel - self.agent_state['speed']
            speed_action = max(min(((len(self.speed_array)-1)/2 + change/self.speed_quantization), len(self.speed_array)-1),0)
            #print(orient_action, speed_action)
            #pdb.set_trace()
            return int(speed_action)*len(self.orientation_array) + int(orient_action)

        return None

    '''
    def return_position(self, ped_id, frame_id):

        try:
            return self.pedestrian_dict[str(ped_id)][str(frame_id)]
        except KeyError:
            while str(frame_id) not in self.pedestrian_dict[str(ped_id)]:
                frame_id -= 1
            return self.pedestrian_dict[str(ped_id)][str(frame_id)]
    '''

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


    def return_position(self, ped_id, frame_id):

        try:
            return copy.deepcopy(self.pedestrian_dict[str(ped_id)][str(frame_id)])
        except KeyError:
            while str(frame_id) not in self.pedestrian_dict[str(ped_id)]:
                frame_id -= 1
            return copy.deepcopy(self.pedestrian_dict[str(ped_id)][str(frame_id)])


    def draw_arrow(self, base_position , next_position, color):

        #base_position = (row,col)
        if np.linalg.norm(base_position - next_position) <= self.step_size*math.sqrt(2):
        #draw the stalk
            arrow_width  = self.cellWidth*.1 #in pixels
            base_pos_pixel = (base_position+.5)
            next_pos_pixel = (next_position+.5)
            #pdb.set_trace()

            #draw the head
            ref_pos = base_pos_pixel+(next_pos_pixel-base_pos_pixel)*.35
            arrow_length = 0.7
            arrow_base = base_pos_pixel
            arrow_end = base_pos_pixel + (next_pos_pixel - base_pos_pixel)* arrow_length

            pygame.draw.line(self.gameDisplay, color, (arrow_base[1], arrow_base[0]),
                             (arrow_end[1], arrow_end[0]), 2)


    def draw_trajectory(self, trajectory=[], color=None):

        #pdb.set_trace()
        arrow_length = 1
        arrow_head_width = 1
        arrow_width = .1
        #denotes the start and end positions of the trajectory

        rad = int(self.cellWidth*.4)
        start_pos=(trajectory[0]['position']+.5)*self.cellWidth
        end_pos=(trajectory[-1]['position']+0.5)*self.cellWidth

        pygame.draw.circle(self.gameDisplay,(0,255,0),
                            (int(start_pos[1]),int(start_pos[0])),
                            rad)

        pygame.draw.circle(self.gameDisplay,(0,0,255),
                    (int(end_pos[1]),int(end_pos[0])),
                    rad)

        for count in range(len(trajectory)-1):
            #pygame.draw.lines(self.gameDisplay,color[counter],False,trajectory_run)
            self.draw_arrow(trajectory[count]['position'],trajectory[count+1]['position'], color)







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
    '''
    feat_drone = DroneFeatureSAM1(step_size=2,
                                  thresh1=10,
                                  thresh2=20)
    
    feat_drone = DroneFeatureRisk_v2(step_size=2,
                                  thresh1=35,
                                  thresh2=60,s
                                  show_agent_persp=True)
    '''
    feat_drone = DroneFeatureRisk_speed(step_size=2,
                                        thresh1=20,
                                        thresh2=30,
                                        show_agent_persp=True)
    #annotation_file = '/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/university_students/annotation/processed/frame_skip_1/students003_processed_corrected.txt'
    annotation_file = '/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/data_zara/annotation/processed/crowds_zara01_processed.txt'
    #annotation_file = None
    world = GridWorldDrone(display=True, 
                        seed=20, obstacles=None, 
                        show_trail=True,
                        is_random=False,
                        annotation_file=annotation_file,
                        subject=None,
                        tick_speed=30, 
                        obs_width=7,
                        step_size=2,
                        agent_width=7,
                        step_reward=0.01,
                        show_comparison=True,
                        show_orientation=True,
                        external_control=True,
                        replace_subject=False, 
                        segment_size=500,
                        consider_heading=True,                      
                        continuous_action=False,
                        rows=576, cols=720, width=20)

    #pf_agent = PFController()
    orient_quant = world.orient_quantization
    orient_div = len(world.orientation_array)
    speed_quant = world.speed_quantization
    speed_div = len(world.speed_array)
    pdb.set_trace()
    agent = SocialForcesController(speed_div, orient_div, orient_quant)


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
        #while world.current_frame < fin_frame:
        while True:
            #action = input()

            action = world.action_space.sample()


            #action = agent.eval_action(state)
            #print("agent state :", world.agent_state)
            #print("current orientation :", world.cur_heading_dir)
            #action_speed = int(input())
            #action_orient = int(input())
            #action = world.action_space.sample()
            #print(action)
            #print(world.agent_state)
            #action = action_speed+(action_orient*7)
            #print('Action :', action)

            state, reward , done, _ = world.step(action)
            #print(reward, done)
            info_collector.collect_information_per_frame(state)
            #print(state['agent_state']['orientation'])
            #feat_drone.overlay_bins(state)
            #print(state)
            #pdb.set_trace()
            feat = feat_drone.extract_features(state)
            #pdb.set_trace()
            #print(feat)
            '''
            if t%100==0:
                world.rollback(10)
                feat_drone.rollback(10, state)
                t=1
            '''
            #feat2 = feat_drone_2.extract_features(state)
            #orientation = feat_drone.extract_features(state)
            '''
            print('Global info:')
            print(feat[0:9].reshape(3, 3))
            print('global info_goal:')
            print(feat[9:18].reshape(3, 3))

            print('Occupancy grid info:')
            print(feat[22:].reshape(window_size, window_size))
            '''
            #pdb.set_trace()

            #print(world.agent_state)
            #print (reward, done)
            t+=1

        #info_collector.collab_end_traj_results()

    #info_collector.collab_end_results()
    
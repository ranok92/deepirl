import sys
import math
import pdb
import itertools
import numpy as np 
from utils import reset_wrapper, step_wrapper
from scipy.ndimage.filters import convolve1d as convolve
import os
import copy
import pygame
##################################################
#*********** feature extracting functions********#


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)



def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    #this function is for [x, y] coordinates,
    #the input vectors v1 and v2 are [row, col]
    v1_corr = np.array([v1[1], -v1[0]])
    v2_corr = np.array([v2[1], -v2[0]])
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def deg_to_rad(deg):

    return deg*np.pi/180

def rad_to_deg(rad):

    return rad*180/np.pi


def get_rot_matrix(theta):
    '''
    returns the rotation matrix given a theta value
    rotates in the counter clockwise direction
    '''
    return np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])



def arange_orientation_info(dim_vector_8):
    #converts the 8 dim vector of orientation to
    # a 9 dim vector, for visulization purposes

    orient_disp_vector = np.zeros(9)
    j = 0
    for i in range(dim_vector_8.shape[0]):

        if i==4:
            j+=1
        orient_disp_vector[j] = dim_vector_8[i]

    return orient_disp_vector


'''
def get_abs_orientation(agent_state, orientation_approximator):
    #returns the current absolute binned orientation of the agent
    #one of the 8 directions. Dim:8 (this is the default case)
    #for the default case, it additionally returns a 9 dimensional vector
    #if no orientation information is provided it returns 4.
    #works for the orientation approximator
    0 1 2
    3   4
    5 6 7 
    ############
    #for other cases, it just returns the orientation.
    #if no orientation information is provided, it returns -1.

    no_of_directions = len(orientation_approximator)
    angle_diff= np.zeros(no_of_directions)

    abs_approx_orientation = None
    if no_of_directions==8: #the default
        #will return the vector only if the orientation_approximator is the default 8-dir one.
        abs_approx_orientation = np.zeros(9)
    else:
        abs_approx_orientation = np.zeros(no_of_directions)
    orientation = agent_state['orientation']
    if orientation is None:
        #straight up
        orientation = 1
    elif np.linalg.norm(orientation)==0:
        if no_of_directions==8:
            orientation = 1
        else:
            orientation = 1
    else:
        for i in range(len(orientation_approximator)):
            #print('The orientation val')
            #print(orientation)
            angle_diff[i] = angle_between(orientation_approximator[i], orientation)

        orientation = np.argmin(angle_diff)
        if no_of_directions == 8:
            if orientation >=4:
                orientation += 1
            abs_approx_orientation[orientation] = 1

            return abs_approx_orientation, orientation

    return abs_approx_orientation, orientation
'''

def get_abs_orientation(agent_state, orientation_approximator):
    '''
    #returns the current absolute binned orientation of the agent
    #one of the 8 directions. Dim:8 (this is the default case)
    #for the default case, it additionally returns a 9 dimensional vector
    #if no orientation information is provided it returns 4.
    #works for the orientation approximator
    0 1 2
    7   3
    6 5 4 
    ############
    #for other cases, it just returns the orientation.
    #if no orientation information is provided, it returns -1.
    '''
    no_of_directions = len(orientation_approximator)
    angle_diff = np.zeros(no_of_directions)

    min_thresh = 0.001
    abs_approx_orientation = None
    if no_of_directions == 8: #the default
        #will return the vector only if the orientation_approximator is the default 8-dir one.
        abs_approx_orientation = np.zeros(9)
    else:
        abs_approx_orientation = np.zeros(no_of_directions)
    orientation = agent_state['orientation']
    if orientation is None:
        #straight up
        orientation = 1

    else:
        for i in range(len(orientation_approximator)):
            #print('The orientation val')
            #print(orientation)
            angle_diff[i] = angle_between(orientation_approximator[i], orientation)

        orientation = np.argmin(angle_diff)
            
        abs_approx_orientation[orientation] = 1

    return abs_approx_orientation, orientation



def get_rel_orientation(prev_frame_info, agent_state, goal_state):
    #returns the relative orientation of the agent with the direction
    #of the goal. 
    #Primarily for use in IRL 
    relative_orientation_vector = np.zeros(4)
    vector_to_goal = goal_state - agent_state['position']
    if prev_frame_info is None:
        agent_orientation = np.array([-1, 0])
    else:
        agent_orientation = agent_state['position'] - prev_frame_info['position']
    diff_in_angle = angle_between(vector_to_goal, agent_orientation)
    #pdb.set_trace()
    if diff_in_angle < np.pi/8:
        rel_orientation = 0

    elif diff_in_angle < np.pi/4 and diff_in_angle >= np.pi/8:
        rel_orientation = 1

    elif diff_in_angle < np.pi*3/4 and diff_in_angle >= np.pi/4:
        rel_orientation = 2

    else:
        rel_orientation = 3

    relative_orientation_vector[rel_orientation] = 1
    return relative_orientation_vector


def get_rel_goal_orientation(orientation_approximator, rel_orient_conv, agent_state, agent_abs_orientation, goal_state):
    #returns the relative orientation of the goal wrt to the agent
    #Dim:8
    
    no_of_directions = len(orientation_approximator)
    angle_diff = np.zeros(no_of_directions)
    relative_orientation_vector = np.zeros(no_of_directions)


    rot_matrix = get_rot_matrix(rel_orient_conv[agent_abs_orientation])

    #translate the point so that the agent sits at the center of the coordinates
    #before rtotation
    vec_to_goal = goal_state - agent_state['position']

    #rotate the coordinates to get the relative coordinates wrt the agent
    rel_coord_goal = np.matmul(rot_matrix, vec_to_goal)

    relative_goal = {}
    relative_goal['orientation'] = rel_coord_goal

    relative_orientation_vector, _ = get_abs_orientation(relative_goal, orientation_approximator)


    '''
    if diff_in_angle < np.pi/4:
        rel_orientation = 0

    elif diff_in_angle < np.pi*3/4 and diff_in_angle >= np.pi/4:
        rel_orientation = 1

    else:
        rel_orientation = 2

    relative_orientation_vector[rel_orientation] = 1 
    '''
    return relative_orientation_vector

def discretize_information(information, information_slabs):
    #given a piece of information(scalar), this function returns the correct 
    #slab in which the information belongs, based on the slab information 
    #information_slab(list)provided
    for i in range(len(information_slabs)-1):

        if information >= information_slabs[i] and information < information_slabs[i+1]:
            return i

    #if does not classify in any information slabs
    return None



def calculate_social_forces(agent_state, obstacle_state, agent_width, obstacle_width,
                            a, b, lambda_val):
    #agent_state and obstacle_state are dictionaries with the following information:
    #position, orientation and speed

    r_i_j = agent_width/2 + obstacle_width/2
    d_i_j = np.linalg.norm(agent_state['position'] - obstacle_state['position'])

#################################################################################
#################################################################################
class DroneFeatureSAM1():
    '''
    Features to put in:
        1. Orientation of the obstacles
        2. Speed of the obstacles
        4. Speed of the agent? 
        N.B. To add speed of the agent, you have to have 
             actions that deal with the speed of the agent.
        5. Density of pedestrian around the agent?
    '''
    '''
    Description of the feature representation:
    Total size : 162 = 9 + 3 + 3 + 3 + 16*9
    Global direction : The direction in which the agent is facing. (9)
    Goal direction : The direction of the goal wrt the agent. (3)
    Inner ring density : The number of people in the inner ring. (3)
    Outer ring density : The number of people in the outer ring. (3)
    Single Bin information : The average speed and orientation of the 
    people in a given bin. (5(speed)+4(orientation))
    Total number of bins : 8x2
    '''

    def __init__(self, thresh1=1, thresh2=2, agent_width=10,
                 obs_width=10, step_size=10, grid_size=10, 
                 show_bins=False
                 ):

        self.agent_width = agent_width
        self.obs_width = obs_width
        self.step_size = step_size
        self.grid_size = grid_size
        #self.prev_frame_info = None
        self.agent_state_history = []
        self.state_rep_size = None

        self.thresh1 = thresh1*step_size
        self.thresh2 = thresh2*step_size
        '''
        orientation approximator format
            0 1 2 
            3   4
            5 6 7
        
        
        self.orientation_approximator = [np.array([-2, -2]), np.array([-2,0]),
                                         np.array([-2, 2]), np.array([0, -2]),
                                         np.array([0, 2]), np.array([2, -2]),
                                         np.array([2, 0]), np.array([2,2])]
        
        '''
        '''
        orientation approximator format
            0 1 2
            7   3
            6 5 4
        '''
        
        self.orientation_approximator = [np.array([-2, -2]), np.array([-2,0]),
                                         np.array([-2, 2]), np.array([0, 2]),
                                         np.array([2, 2]), np.array([2, 0]),
                                         np.array([2, -2]), np.array([0, -2])]
        
        '''
            0
        3       1
            2
        '''
        self.orientation_approximator_4 = [np.array([-2, 0]), np.array([0, 2]),
                                           np.array([2, 0]), np.array([0, -2])]
        
        '''
        self.rel_orient_conv = [7*np.pi/4, 0,
                                np.pi/4, 6*np.pi/4,
                                np.pi/2, 5*np.pi/4,
                                np.pi, 3*np.pi/4]
        '''

        self.rel_orient_conv = [7*np.pi/4, 0,
                                1*np.pi/4, 2*np.pi/4,
                                3*np.pi/4, 4*np.pi/4,
                                5*np.pi/4, 6*np.pi/4 ]

        '''
        self.rel_orient_conv = [np.pi/4, 0, 7*np.pi/4,
                                2*np.pi/4, 6*np.pi/4,
                                3*np.pi/4, 4*np.pi/4, 5*np.pi/4]
        '''
        self.speed_divisions = [0, 1, 2, 5]
        self.inner_ring_density_division = [0, 1, 2, 4]
        self.outer_ring_density_division = [0, 3, 5, 7]
        self.show_bins = show_bins
        #self.bins is a dictionary, with keys containing the id of the bins and 
        #corresponding to each bin is a list containing the obstacles 
        #present in the bin

        self.bins = {}
        for i in range(16):
            self.bins[str(i)] = []

        self.state_dictionary = {}
        self.state_str_arr_dict = {}
        self.inv_state_dictionary = {}
        self.hash_variable_list = []

        self.num_of_speed_blocks = 3
        self.num_of_orient_blocks = 4

        #state rep size = 16*8+9+3+3

        #state rep size = 9+9+4+16*8+3+3
        self.state_rep_size = 131
        self.generate_hash_variable()
        #self.generate_state_dictionary()
        #print('Done!')


    def smooth_state(self, state):
        '''
        A smoothing function for a given state
        depending how the feature extractor is depicting the state.
        Each feature extractor should ideally have one.

        input - state(numpy)
        output - a smoothed version of the state vector(numpy) based on how the 
        state feature has been designed in the first place
        '''
        return state


    def generate_hash_variable(self):
        '''
        The hash variable basically is an array of the size of the current state. 
        This creates an array of the following format:
        [. . .  16 8 4 2 1] and so on.
        '''
        self.hash_variable_list = []
        for i in range(self.state_rep_size-1, -1, -1):
    
            self.hash_variable_list.append((int(math.pow(2, self.state_rep_size-1-i))))
        

    def recover_state_from_hash_value(self, hash_value):

        size = self.state_rep_size
        state_val = np.zeros(size)
        i = 0
        while hash_value > 0:
            state_val[i] = int(hash_value)%2
            hash_value = math.floor((hash_value)//2)
            i += 1

        return state_val


    def hash_function(self, state):

        hash_value = 0
        size = len(self.hash_variable_list)
        for i in range(size):
            hash_value += int(self.hash_variable_list[i]*state[i])

        return hash_value

    def get_info_from_state(self, state):
        #read information from the state
        agent_state = state['agent_state']
        goal_state = state['goal_state']
        obstacles = state['obstacles']

        return agent_state, goal_state, obstacles



    def get_relative_coordinates(self, ):
        #adjusts the coordinates of the obstacles based on the current
        #absolute orientation of the agent. 

        return 0



    def populate_orientation_bin(self, agent_orientation_val, agent_state, obs_state_list):
        '''
        #given an obstacle, the agent state and orientation,
        #populates the self.bins dictionary with the appropriate obstacles
        #self.bins is a dictionary where against each key of the dictionary
        #is a list of obstacles that are present in that particular bin
        Bin informations:
            Bins from the inner ring 0:7
            Bins from the outer ring 8:15
            Bin value in each of the ring is based on the orientation_approximator

        '''
        #for debugging purposes
        #print('***INSIDE populate_orientation_bin ***')
        
        for obs_state in obs_state_list:

            distance = np.linalg.norm(obs_state['position'] - agent_state['position'])
            #pdb.set_trace()
            if obs_state['orientation'] is not None:
                obs_orientation_ref_point = obs_state['position'] + obs_state['orientation']
            else:
                obs_orientation_ref_point = obs_state['position']

            ring_1 = False
            ring_2 = False
            if distance < self.thresh2:
                #classify obs as considerable
                #check the distance
                temp_obs = {}
                if distance < self.thresh1:
                    #classify obstacle in the inner ring
                    ring_1 = True
                else:
                    #classify obstacle in the outer ring
                    ring_2 = True

                #check for the orientation
                #obtain relative orientation
                #get the relative coordinates
                rot_matrix = get_rot_matrix(deg_to_rad(agent_orientation_val))

                #translate the point so that the agent sits at the center of the coordinates
                #before rtotation
                vec_to_obs = obs_state['position'] - agent_state['position']
                vec_to_orient_ref = obs_orientation_ref_point - agent_state['position']

                #rotate the coordinates to get the relative coordinates wrt the agent
                rel_coord_obs = np.matmul(rot_matrix, vec_to_obs)
                rel_coord_orient_ref = np.matmul(rot_matrix, vec_to_orient_ref)

                angle_diff = np.zeros(8)
                for i in range(len(self.orientation_approximator)):
                    #print('The orientation val')
                    #print(orientation)
                    '''
                    print("***")
                    print(self.orientation_approximator[i])
                    print("---")
                    print(rel_coord_obs)
                    print("+++")
                    print(vec_to_obs)
                    print("####")
                    print(rot_matrix)
                    print("+++")
                    print(agent_orientation_val)
                    print("***")
                    '''
                    angle_diff[i] = angle_between(self.orientation_approximator[i], rel_coord_obs)            
                bin_val = np.argmin(angle_diff)

                if ring_2:
                    bin_val += 8

                #orientation of the obstacle needs to be changed as it will change with the 
                #change in the relative angle. No need to change the speed.
                
                temp_obs['orientation'] = rel_coord_orient_ref - rel_coord_obs
                temp_obs['position'] = rel_coord_obs
                temp_obs['speed'] = obs_state['speed']

                '''
                #******for debugging purposes
                print('The current agent heading direction :', agent_orientation_val)
                print('Change in position : before :{}, after :{}'.format(obs_state['position'],
                                                                          temp_obs['position']))
                print('Change in orientation : before :{}, after : {}'.format(obs_state['orientation'],
                                                                              temp_obs['orientation']))
                print('Change in speed : before :{}, after :{}'.format(obs_state['speed'], temp_obs['speed']))
                #pdb.set_trace()
                '''

                self.bins[str(bin_val)].append(temp_obs)
        #if the obstacle does not classify to be considered
    
    def overlay_bins(self, state):

        #a visualizing tool to debug if the binning is being done properly
        #draws the bins on the game surface for a visual inspection of the 
        #classification of the obstacles in their respective bins
        #pdb.set_trace()
        self.orientation_approximator
        #draw inner ring
        #pdb.set_trace()
        center =  np.array([int(state['agent_state']['position'][1]),
                  int(state['agent_state']['position'][0])])
        pygame.draw.circle(pygame.display.get_surface(), (0,0,0), center, self.thresh1,2) 
        #draw outer ring 
        pygame.draw.circle(pygame.display.get_surface(), (0,0,0), center, self.thresh2,2)

        line_start_point = np.array([0, -self.thresh2])
        line_end_point = np.array([0,self.thresh2])
        for i in range(8):
            #draw the lines
            rot_matrix = get_rot_matrix(self.rel_orient_conv[i])
            cur_line_start = np.matmul(rot_matrix, line_start_point) + center
            cur_line_end = np.matmul(rot_matrix, line_end_point) + center
            #pdb.set_trace()
            pygame.draw.line(pygame.display.get_surface(), (0,0,0), cur_line_start,
                             cur_line_end,2)
        pygame.display.update()
        #pdb.set_trace()
    


    def compute_bin_info(self):
        #given self.bins populated with the obstacles,
        #computes the average relative orientation and speed for all the bins
        sam_vector = np.zeros([16,len(self.speed_divisions)-1+len(self.orientation_approximator_4)])
        density_inner_ring = np.zeros(3)
        inner_ring_count = 0
        density_outer_ring = np.zeros(3)
        outer_ring_count = 0
        for i in range(len(self.bins.keys())):
            avg_speed = 0
            avg_orientation = np.zeros(2)

            speed_bin = np.zeros(len(self.speed_divisions)-1)
            orientation_bin = np.zeros(len(self.orientation_approximator_4))

            total_obs = len(self.bins[str(i)])

            for j in range(total_obs):
                obs = self.bins[str(i)][j]
                if obs['speed'] is not None:
                    avg_speed += np.linalg.norm(obs['speed'])

                if obs['orientation'] is not None:
                    avg_orientation += obs['orientation']

                if i < 8:
                    inner_ring_count+=1
                else:
                    outer_ring_count+=1
                
            #if obs['speed'] is not None:
            if total_obs > 0:
                avg_speed /= total_obs
                speed_bin_index = discretize_information(avg_speed, self.speed_divisions)
                speed_bin[speed_bin_index] = 1

                #if obs['orientation'] is not None:
                new_obs = {'orientation': avg_orientation}
                _, avg_orientation = get_abs_orientation(new_obs, self.orientation_approximator_4)
                #print('the avg orientation :', avg_orientation)
                orientation_bin[avg_orientation] = 1

                #based on the obtained average speed and orientation bin them
                #print('Avg speed :', avg_speed, 'Speed bin :',speed_bin)
                #print('Avg orientation :', avg_orientation, 'Orientation bin :', orientation_bin)
            sam_vector[i][:] = np.concatenate((speed_bin, orientation_bin))

        density_inner_ring[discretize_information(inner_ring_count, 
                                                  self.inner_ring_density_division)] = 1
        density_outer_ring[discretize_information(outer_ring_count,
                                                  self.outer_ring_density_division)] = 1

        return sam_vector, density_inner_ring, density_outer_ring



    def compute_social_force(self):
        #computes the social force value at a given time(optional)

        return 0 



    def extract_features(self, state):
        #getting everything to come together to extract the features

        agent_state, goal_state, obstacles = self.get_info_from_state(state)
        abs_approx_orientation, agent_orientation_index = get_abs_orientation(agent_state, self.orientation_approximator)

        agent_orientation_angle = state['agent_head_dir']

        #print('The orientation :')
        #print(abs_approx_orientation.reshape(3,3))
        if len(self.agent_state_history) > 0:
            prev_frame_info = self.agent_state_history[-1]
        else:
            prev_frame_info = None
        relative_orientation = get_rel_orientation(prev_frame_info, agent_state, goal_state)
        relative_orientation_goal = get_rel_goal_orientation(self.orientation_approximator,
                                                   self.rel_orient_conv,
                                                   agent_state, 
                                                   agent_orientation_index,
                                                   goal_state)

        #print('The absolute approx orientation :', abs_approx_orientation)
        ##print('The relative orientation', relative_orientation)

        #empty bins before populating
        for i in range(16):
            self.bins[str(i)] = []


        #print('Here')
        self.populate_orientation_bin(agent_orientation_angle, agent_state, obstacles)
        #pdb.set_trace()
        sam_vector, inner_ring_density, outer_ring_density = self.compute_bin_info()

        extracted_feature = np.concatenate((#abs_approx_orientation,
                                            relative_orientation_goal,
                                            relative_orientation,
                                            np.reshape(sam_vector,(-1)),
                                            inner_ring_density,
                                            outer_ring_density))
        '''
        flag = False
        for i in range(16):
            if len(self.bins[str(i)]) > 0:
                flag = True
        if flag:    
            pdb.set_trace()
            pass
        '''
        self.agent_state_history.append(copy.deepcopy(state['agent_state']))

        return reset_wrapper(extracted_feature)


    def rollback(self, frames, state):

        if frames > len(self.agent_state_history):
            print('Trying to rollback more than it has seen!!!')
        else:
            for i in range(1, frames+1):
                if len(self.agent_state_history) > 0:
                    self.agent_state_history.pop(-1)

        return self.extract_features(state)


    def reset(self):

        self.agent_state_history = []


class DroneFeatureMinimal(DroneFeatureSAM1):

    def __init__(self, thresh1=1, thresh2=2,
                agent_width=10, step_size=10,
                obs_width=10, goal_size=10, show_bins=False
                ):
        super().__init__(thresh1=thresh1,
                         thresh2=thresh2,
                         agent_width=agent_width,
                         step_size=step_size,
                         grid_size=goal_size,
                         show_bins=show_bins,
                         obs_width=obs_width
                         )
        self.thresh_speed = 0.5
        self.state_rep_size = 50


    def compute_bin_info(self):
        '''
        The minimal version ditches the huge detailed vectors of information
        for something more succient. It returns 3/2 dimensional vector telling
        how likely the pedestrians in the bin is to interfere with the robot.

        Likelihood of collision is calculated as follows:
            Low : if the pedestrians of the bin are moving away from the agent
            High : if the pedestrians are quick and moving towards the agent
            Med : Anything that does not fall in this category
        '''
        collision_information = np.zeros((len(self.bins.keys()), 3))
        for i in range(len(self.bins.keys())):
            #for each bin
            current_danger_level = 0
            for ped in range(len(self.bins[str(i)])):
                #for each pedestrian
                #pdb.set_trace()
                coll = self.compute_collision_likelihood(self.bins[str(i)][ped])
                if coll>current_danger_level:
                    current_danger_level = coll

            collision_information[i,current_danger_level] = 1
        
        if np.sum(collision_information[:,1])>0 or np.sum(collision_information[:,2]) > 0:

            for i in range(collision_information.shape[0]):
                print('Bin no :', i, ', collision_info : ', collision_information[i,:])

            #pdb.set_trace()
        
        return collision_information



    def compute_collision_likelihood(self, pedestrian):
        '''
        collision prob: High : 2, med : 1, low : 0
        '''
        collision_prob = 0
        pos_vector = np.array([0,0]) - pedestrian['position']
        orientation = pedestrian['orientation']
        ang = angle_between(pos_vector, orientation)

        #highest prob
        if ang < np.pi/8:
            if np.linalg.norm(pedestrian['orientation']) > self.thresh_speed:
                collision_prob = 2
        #lowest prob
        elif ang > np.pi/8 or pedestrian['speed'] == 0:
            collision_prob = 0
        #somewhere in between
        else:
            collision_prob = 1

        return collision_prob


    def extract_features(self, state):

        agent_state, goal_state, obstacles = self.get_info_from_state(state)
        abs_approx_orientation, agent_orientation_index = get_abs_orientation(agent_state, self.orientation_approximator)
        
        if len(self.agent_state_history) > 0:
            prev_frame_info = self.agent_state_history[-1]
        else:
            prev_frame_info = None

        relative_orientation = get_rel_orientation(prev_frame_info, agent_state, goal_state)

        for i in range(16):
            self.bins[str(i)] = []

        self.populate_orientation_bin(agent_orientation_index, agent_state, obstacles)

        collision_info = self.compute_bin_info()

        self.agent_state_history.append(copy.deepcopy(state['agent_state']))
        #pdb.set_trace()

        #return reset_wrapper(extracted_feature)


        return None



class DroneFeatureOccup(DroneFeatureSAM1):

    def __init__(self, thresh1=1, thresh2=2,
            agent_width=10, step_size=10,
            obs_width=10, grid_size=10, 
            show_bins=False, window_size=5
            ):
        super().__init__(thresh1=thresh1,
                         thresh2=thresh2,
                         agent_width=agent_width,
                         step_size=step_size,
                         grid_size=grid_size,
                         show_bins=show_bins,
                         obs_width=obs_width
                         )
        self.window_size = window_size
        self.thresh_speed = 0.5
        self.state_rep_size = window_size**2 + 22
        self.thresh2 = (step_size * window_size)/2


    def check_overlap(self,temp_pos,obs_pos):
        #if true, that means there is an overlap
        boundary = None
        if self.grid_size >= self.agent_width:
            boundary = self.grid_size/2
        else:
            boundary = self.agent_width/2

        distance_to_maintain = boundary+(self.obs_width/2)
        #pdb.set_trace()
        if abs(temp_pos[0] - obs_pos[0]) < distance_to_maintain and abs(temp_pos[1] - obs_pos[1]) < distance_to_maintain:

            return True
        else:
            return False


    def block_to_arrpos(self,r,c):

        a = (self.window_size**2-1)/2
        b = self.window_size
        pos = a+(b*r)+c
        return int(pos)

    '''
    def overlay_grid(self, pygame_surface, state):


        center =  np.array([int(state['agent_state']['position'][1]),
                  int(state['agent_state']['position'][0])])


        window_rows = window_cols = self.window_size
        line_orient = ['hor', 'ver']

        grid_width = self.step_size

        start_point = center - np.array([window_size/2 ])
        for orient in line_orient:
            for i in range(window_size):

                start_point = 
    '''


    def compute_bin_info(self):

        obstacles = []
        #create a obstacle list from the self.bins
        for bin_key in self.bins.keys():

            for obs in self.bins[bin_key]:
                obstacles.append(obs)

        window_rows = window_cols = self.window_size
        row_start = int((window_rows-1)/2)
        col_start = int((window_cols-1)/2)

        local_occup_grid = np.zeros(self.window_size**2)
        agent_pos = np.array([0,0])

        for i in range(len(obstacles)):

            #as of now this just measures the distance from the center of the obstacle
            #this distance has to be measured from the circumferance of the obstacle

            #new method, simulate overlap for each of the neighbouring places
            #for each of the obstacles
            obs_pos = obstacles[i]['position']
            obs_width = self.obs_width
            for r in range(-row_start,row_start+1,1):
                for c in range(-col_start,col_start+1,1):
                    #c = x and r = y
                    #pdb.set_trace()
                    temp_pos = np.asarray([agent_pos[0] + r*self.step_size, 
                                agent_pos[1] + c*self.step_size])
                    if self.check_overlap(temp_pos,obs_pos):
                        pos = self.block_to_arrpos(r,c)

                        local_occup_grid[pos] = 1

        return local_occup_grid


    def extract_features(self, state):
        #getting everything to come together to extract the features

        agent_state, goal_state, obstacles = self.get_info_from_state(state)
        abs_approx_orientation, agent_orientation_index = get_abs_orientation(agent_state, self.orientation_approximator)

        #print('The orientation :')
        #print(abs_approx_orientation.reshape(3,3))

        if len(self.agent_state_history) > 0:
            prev_frame_info = self.agent_state_history[-1]
        else:
            prev_frame_info = None

        relative_orientation = get_rel_orientation(prev_frame_info, agent_state, goal_state)
        relative_orientation_goal = get_rel_goal_orientation(self.orientation_approximator,
                                                   self.rel_orient_conv,
                                                   agent_state, 
                                                   agent_orientation_index,
                                                   goal_state)

        #print('The absolute approx orientation :', abs_approx_orientation)
        ##print('The relative orientation', relative_orientation)

        #empty bins before populating
        for i in range(16):
            self.bins[str(i)] = []


        #print('Here')
        self.populate_orientation_bin(agent_orientation_index, agent_state, obstacles)
        #pdb.set_trace()
        local_occup_grid = self.compute_bin_info()

        extracted_feature = np.concatenate((abs_approx_orientation,
                                            relative_orientation_goal,
                                            relative_orientation,
                                            local_occup_grid))

        self.agent_state_history.append(copy.deepcopy(state['agent_state']))

        return reset_wrapper(extracted_feature)



class DroneFeatureRisk(DroneFeatureSAM1):

    def __init__(self, thresh1=1, thresh2=2,
                 agent_width=10, step_size=10,
                 obs_width=10, grid_size=10,
                 show_bins=False, 
                 show_agent_persp=False
                 ):

        super().__init__(thresh1=thresh1,
                         thresh2=thresh2,
                         agent_width=agent_width,
                         step_size=step_size,
                         grid_size=grid_size,
                         show_bins=show_bins,
                         obs_width=obs_width
                         )

        self.rel_speed_divisions = [-1,0,1]
        self.rel_distance_divisions = [ 1, 3, 5]
        '''
        relative goal : 9
        relative step : 4
        risk information for 16 bins : 16*3
        '''
        self.state_rep_size = 9+4+16*3
        self.generate_hash_variable()

        self.show_agent_persp = show_agent_persp
        self.init_surface = False
        self.orig_disp_size_row = None
        self.orig_disp_size_col = None
    '''
    def show_agent_view(self, agent_orientation_val, agent_state, pygame_surface):

        #draw the agent


        #draw the bins


        #draw the obstacles
        if agent_orientation_val > 4:
            agent_orientation_val -= 1

        rot_matrix = get_rot_matrix(self.rel_orient_conv[agent_orientation_val])

        if agent_state['orientation'] is None:
            agent_state['orientation'] = np.array([1, 0])

        rotated_agent_orientation = np.matmul(rot_matrix, agent_state['orientation'])
        for key in self.bins.keys():

            for obs in obs_list:

                rel_orient = obs['orientation'] - rotated_agent_orientation
    '''

    def get_change_in_orientation(self, cur_agent_orientation):

        #cur_agent_orientation is a 2d array [row, col]
        prev_agent_orient = None
        change_vector = np.zeros(5)
        if len(self.agent_state_history) > 0:
            prev_frame_info = self.agent_state_history[-1]
        else:
            prev_frame_info = None
        if prev_frame_info is not None and cur_agent_orientation is not None:
            prev_agent_orient = prev_frame_info['orientation']
            angle_diffs = np.array([0, np.pi/4, np.pi/2, np.pi*3/4, np.pi])
            diff_in_angle = angle_between(prev_agent_orient, cur_agent_orientation)
            index = np.argmin(np.abs(angle_diffs - diff_in_angle))

            #print('Prev orientation :', prev_agent_orient)
            #print('cur_agent_orientation :', cur_agent_orientation)
        else:
            index = 0

        #print('Index selected :', index)
        #pdb.set_trace()
        change_vector[index] = 1
        return change_vector



    def compute_bin_info(self, agent_orientation_val, agent_state, pygame_surface=None):



        risk_vector = np.zeros((len(self.bins.keys()),3))
        #rotate the agent's orientation to match that of the obstacles
        thresh_value = self.agent_width/2 + self.obs_width/2 + self.step_size
        thresh_value += self.agent_width #padding
        #pdb.set_trace()

        rot_matrix = get_rot_matrix(-agent_orientation_val)
        if agent_state['orientation'] is None:
            agent_state['orientation'] = np.array([1, 0])
        rotated_agent_orientation = np.matmul(rot_matrix, agent_state['orientation'])

        '''
        #***for debugging purposes****
        print('The orientation val :', agent_orientation_val)
        print('Agent orientation val :', agent_orientation_val)
        print('Rotated agent orientation :', rotated_agent_orientation)
        #*****************************
        '''

        pad = 80
        mag = 1 #magnification of the orientation lines
        
        ################################
        #code for the agent view
        #make changes in the game display accordingly
        #this is a onetime thing
        if self.show_agent_persp and not self.init_surface:
            #draw the bins
            self.orig_disp_size_col, self.orig_disp_size_row = pygame.display.get_surface().get_size()
            pygame.display.set_mode((self.orig_disp_size_col+self.thresh2*2+pad,
                                     self.orig_disp_size_row))
            self.init_surface = True

        #add the agent view, refreshed every step
        if self.show_agent_persp:

            #center is in (row, col) format
            center = (self.orig_disp_size_row/2, 
                      self.orig_disp_size_col+ self.thresh2+pad/2)

            dummy_state = {'agent_state':{}}
            dummy_state['agent_state']['position'] = center
            side = self.thresh2*2 + pad/2
      

            #clear and re-draw the primary agent_view rectangle
            pygame.display.get_surface().fill((255, 255, 255), ((self.orig_disp_size_col, 0),
                            (self.thresh2*2+pad, self.orig_disp_size_row)))

            pygame.draw.line(pygame.display.get_surface(), (0, 0, 0),
                            (self.orig_disp_size_col,0),(self.orig_disp_size_col,self.orig_disp_size_row),
                            3)
            pygame.draw.rect(pygame.display.get_surface(), (0, 0, 0), 
                            ((center[1]-side/2, center[0]-side/2),
                            (side, side)), 4)
            #draw the cicles
            #spdb.set_trace()
            self.overlay_bins(dummy_state)
            #draw the agent
            pygame.draw.rect(pygame.display.get_surface(), (0,0,0), 
                            [center[1]-self.agent_width/2, center[0]-self.agent_width/2, 
                             self.agent_width, self.agent_width])


            #draw the orientation
            '''
            pygame.draw.line(pygame.display.get_surface(), (0,0,0), (center[1], center[0]),
                             ((center[1]+rotated_agent_orientation[1]*mag), (center[0]+rotated_agent_orientation[0]*mag)
                             ), 2)
            '''
            pygame.display.update()
        
        #################################

        for key in self.bins.keys():

            risk_val = 0
            obs_list = self.bins[key]

            #print('Bin :', key)
            for obs in obs_list:

                #relative orientation of the obstacle wrt the agent
                #print('Obs information wrt pygame :', obs['orientation'])
                rel_orient = obs['orientation'] - rotated_agent_orientation
                #print('Relative orientation :', rel_orient)
                #relative position of the agent wrt the obstacle
                rel_dist = -obs['position']

                ang = angle_between(rel_orient, rel_dist)
                
                #if np.linalg.norm(rel_dist) < (self.agent_width+self.obs_width)/2+self.step_size:
                if np.linalg.norm(rel_dist) < (self.agent_width/math.sqrt(2) + self.obs_width/math.sqrt(2) + self.step_size*math.sqrt(2)):
                    risk_val = max(risk_val, 2)
                
                if ang < np.pi/4 and math.tan(ang)*np.linalg.norm(rel_dist) < thresh_value:
                #if ang < np.pi/8:
                    #print('Moving towards')
                    #high risk
                    #adding to it, the rel_distance in both row and 
                    #col should be less than the sum(agent_width/2+obs_width/2)
                    risk_val = max(risk_val, 2) 
                elif ang < np.pi/2:
                    #print('Moving away')
                    #medium risk
                    risk_val = max(risk_val, 1)
                else:
                    #low risk
                    pass


                if self.show_agent_persp:
                    #determine the color of the obstacle based on the risk it poses
                    if risk_val==0:
                        color_val = (0, 255, 0)
                    if risk_val==1:
                        color_val = (0, 0, 255)
                    if risk_val==2:
                        color_val = (255, 0, 0)

                    #draw the obstacle in the agent persepective window
                    shifted_obs_pos = (center[0]+obs['position'][0], center[1]+obs['position'][1])
                    pygame.draw.rect(pygame.display.get_surface(), color_val, 
                                    [shifted_obs_pos[1]-self.obs_width/2, 
                                     shifted_obs_pos[0]-self.obs_width/2,
                                     self.obs_width, self.obs_width])

                    #draw the obstacle orientation in the agent perspective window

                    pygame.draw.line(pygame.display.get_surface(), color_val,
                                    (shifted_obs_pos[1], shifted_obs_pos[0]),
                                    (shifted_obs_pos[1]+rel_orient[1]*mag,
                                     shifted_obs_pos[0]+rel_orient[0]*mag),
                                    2
                                    )

                    pygame.display.update()

                #
            #pdb.set_trace()

            risk_vector[int(key)][risk_val] = 1

        return risk_vector


    def extract_features(self, state):

        agent_state, goal_state, obstacles = self.get_info_from_state(state)
        abs_approx_orientation, agent_orientation_index = get_abs_orientation(agent_state, self.orientation_approximator)

        if len(self.agent_state_history) > 0:
            prev_frame_info = self.agent_state_history[-1]
        else:
            prev_frame_info = None

        relative_orientation = get_rel_orientation(prev_frame_info, agent_state, goal_state)
        relative_orientation_goal = get_rel_goal_orientation(self.orientation_approximator,
                                                   self.rel_orient_conv,
                                                   agent_state, 
                                                   agent_orientation_index,
                                                   goal_state)

        for i in range(16):
            self.bins[str(i)] = []

        #print('absolute orientation :', abs_approx_orientation.reshape((3,3)))
        #print('relative orientation :', relative_orientation_goal.reshape((3,3)))
        self.populate_orientation_bin(agent_orientation_index, agent_state, obstacles)

        collision_info = self.compute_bin_info(agent_orientation_index, agent_state)

        self.agent_state_history.append(copy.deepcopy(state['agent_state']))
        extracted_feature = np.concatenate((relative_orientation,
                                            relative_orientation_goal,
                                            collision_info.reshape((-1))
                                            ))
        #spdb.set_trace()
        return reset_wrapper(extracted_feature)


class DroneFeatureRisk_v2(DroneFeatureRisk):

    def __init__(self, thresh1=1, thresh2=2,
                 agent_width=10, step_size=10,
                 obs_width=10, grid_size=10,
                 show_bins=False, 
                 show_agent_persp=False
                 ):

        super().__init__(thresh1=thresh1,
                         thresh2=thresh2,
                         agent_width=agent_width,
                         obs_width=obs_width,
                         step_size=step_size,
                         grid_size=grid_size,
                         show_bins=show_bins,
                         show_agent_persp=show_agent_persp
                         )

        #change the state representation size accordingly
        '''
        relative_orientation 9
        relative_orientation_goal 4 
        change_in_orientation 5
        collision_info 48
        '''
        self.state_rep_size = 9+4+5+16*3
        self.generate_hash_variable()


    def extract_features(self, state):
        '''
        the parameter ignore_cur_state, if set to true indicates that this is a part of a rollback play.

        '''
        agent_state, goal_state, obstacles = self.get_info_from_state(state)
        abs_approx_orientation, agent_orientation_index = get_abs_orientation(agent_state, self.orientation_approximator)


        agent_orientation_angle = state['agent_head_dir']
        #print('Current heading direction :', agent_orientation_angle)
        if len(self.agent_state_history) > 0:
            prev_frame_info = self.agent_state_history[-1]
        else:
            prev_frame_info = None

        relative_orientation = get_rel_orientation(prev_frame_info, agent_state, goal_state)
        relative_orientation_goal = get_rel_goal_orientation(self.orientation_approximator,
                                                   self.rel_orient_conv,
                                                   agent_state, 
                                                   agent_orientation_index,
                                                   goal_state)

        change_in_orientation = self.get_change_in_orientation(state['agent_state']['orientation'])

        for i in range(16):
            self.bins[str(i)] = []

        #print('absolute orientation :', abs_approx_orientation.reshape((3,3)))
        #print('relative orientation :', relative_orientation_goal.reshape((3,3)))
        self.populate_orientation_bin(agent_orientation_angle, agent_state, obstacles)

        collision_info = self.compute_bin_info(agent_orientation_angle, agent_state)

        
        self.agent_state_history.append(copy.deepcopy(state['agent_state']))
        
        extracted_feature = np.concatenate((relative_orientation,
                                            relative_orientation_goal,
                                            change_in_orientation,
                                            collision_info.reshape((-1))
                                            ))
        '''
        #***debugging block*****#
        print('Relative orientation :', relative_orientation)
        print('Relative orientation goal :', relative_orientation_goal.reshape(3,3))
        print('Change in orientation :', change_in_orientation)
        pdb.set_trace()
        #****end block****#
        '''
        return reset_wrapper(extracted_feature)




class DroneFeatureRisk_speed(DroneFeatureRisk):

    def __init__(self, thresh1=1, thresh2=2,
                 agent_width=10, step_size=10,
                 obs_width=10, grid_size=10,
                 show_bins=False, 
                 max_speed=2,
                 show_agent_persp=False,
                 return_tensor=False
                 ):

        super().__init__(thresh1=thresh1,
                         thresh2=thresh2,
                         agent_width=agent_width,
                         obs_width=obs_width,
                         step_size=step_size,
                         grid_size=grid_size,
                         show_bins=show_bins,
                         show_agent_persp=show_agent_persp
                         )

        #change the state representation size accordingly
        '''
        relative_orientation 9
        relative_orientation_goal 4 
        change_in_orientation 5
        collision_info 48
        speed_info 6
        '''
        self.state_rep_size = 4+9+5+16*3+6
        self.max_speed = max_speed
        self.speed_divisions = 6
        self.generate_hash_variable()
        self.return_tensor = return_tensor


    def smooth_state(self, state):

        '''
        Drone feature risk has 5 parts;
            relative orientation
            relative orientation goal
            change in orientation 
            collision info
            speed info

        Divide the state vector into the above define parts and each of the
        cases separately. Finally concatenate to get the final smoothened state
        '''

        smoothing_kernel_general = np.array([0.1, .8, .1])
        
        #relative orientation : asymmetric features, so kind of hacky
        rel_orient = state[0:4]
        if rel_orient[0]==1:
            smoothing_kernel = np.array([.9, .1])#.8, .2
        if rel_orient[1]==1:
            smoothing_kernel = np.array([.1, .9, 0]) #.2, .8
        if rel_orient[2]==1:
            smoothing_kernel = np.array([0.05, .9, 0.05]) #.05, .9, .05
        if rel_orient[3]==1:
            smoothing_kernel = np.array([0.1, .9, 0]) #[.1, .9, 0]

        rel_orient_smooth = np.convolve(rel_orient, smoothing_kernel, 'same')

        #relative_orientation_goal
        #just take the first 8 and do the convolve
        relative_orientation_goal = state[4:4+8].astype(np.float)
        relative_orientation_goal_full = state[4:4+9]
        smoothing_kernel = smoothing_kernel_general
        relative_orientation_goal_smooth = convolve(relative_orientation_goal, 
                                                    smoothing_kernel, mode='wrap')
        relative_orientation_goal_smooth_9 = np.zeros(9)
        relative_orientation_goal_smooth_9[0:8] = relative_orientation_goal_smooth
        #change in orientation
        #no wrap this time
        change_in_orientation = state[13:13+5]
        smoothing_kernel = smoothing_kernel_general
        change_in_orientation_smooth = np.convolve(change_in_orientation,
                                                  smoothing_kernel,'same')
        #normalize the weights so that the sum remains 1
        change_in_orientation_smooth = change_in_orientation_smooth/np.sum(change_in_orientation_smooth)


        #local bin information
        #bin information comes in a matrix of size 16 * 3 
        #the convolution will happen in axis = 1
        #bin information are in two concentric cicle
        #so have to separate the two circles before smoothing
        risk_info = state[18:18+48].reshape([16,3]).astype(np.float)
        risk_info_inner_circle = risk_info[0:8,:]
        risk_info_outer_circle = risk_info[8:,:]
        smoothing_kernel = smoothing_kernel_general
        #smooth the risk values spatially. ie. moderate risk in a bin will be
        #smoothened to moderate risk to nearby bins. Moderate risk will not be 
        #smoothened to low or high risk
        risk_info_inner_circle_smooth = np.zeros(risk_info_inner_circle.shape)
        risk_info_outer_circle_smooth = np.zeros(risk_info_outer_circle.shape)

        #going through each of the columns (ie the risk levels)
        #the smoothing does not smooth over the risk levels
        #ie. high risk at a bin never smoothens to be a medium or low risk
        #in someother bin.
        for i in range(risk_info_inner_circle.shape[1]):
            risk_info_part = risk_info_inner_circle[:,i]
            risk_info_part_smooth = convolve(risk_info_part, smoothing_kernel, mode='wrap')
            risk_info_inner_circle_smooth[:,i] = risk_info_part_smooth

        for i in range(risk_info_outer_circle.shape[1]):
            risk_info_part = risk_info_outer_circle[:,i]
            risk_info_part_smooth = convolve(risk_info_part, smoothing_kernel, mode='wrap')
            risk_info_outer_circle_smooth[:,i] = risk_info_part_smooth

        #speed information
        #no wrap in the smoothing function
        speed_information = state[-6:]
        smoothing_kernel = smoothing_kernel_general
        speed_information_smooth = np.convolve(speed_information, smoothing_kernel, 'same')
        #normalize the weights so that the sum remains 1
        speed_information_smooth = speed_information_smooth/np.sum(speed_information_smooth)
        
        #********* for debugging purposes *********
        '''
        print('State information :')
        print ("relative orientation")
        print(rel_orient, " ", rel_orient_smooth)
        
        print("relative_orientation_goal")
        print(relative_orientation_goal_full, "  " , relative_orientation_goal_smooth_9)

        print("change in orienatation")
        print(change_in_orientation, "  ", change_in_orientation_smooth)

        print("risk information")
        print("inner circle")
        print(np.c_[risk_info_inner_circle, risk_info_inner_circle_smooth])

        print("outer circle")
        print(np.c_[risk_info_outer_circle, risk_info_outer_circle_smooth])

        print("speed information")
        print(speed_information, '  ', speed_information_smooth)
        pdb.set_trace()
        #*******************************************
        '''
        return np.concatenate((rel_orient_smooth,
                              relative_orientation_goal_smooth_9,
                              change_in_orientation_smooth,
                              risk_info_inner_circle_smooth.reshape((-1)),
                              risk_info_outer_circle_smooth.reshape((-1)),
                              speed_information_smooth))




    def get_speed_info(self, agent_state):

        speed_info = np.zeros(self.speed_divisions)
        cur_speed = agent_state['speed']

        if cur_speed is None:
            cur_speed = 0

        if cur_speed >= self.max_speed:
            cur_speed = self.max_speed-0.001

        quantization = self.max_speed/self.speed_divisions
        speed_info[int(cur_speed/quantization)] = 1
        return speed_info


    def extract_features(self, state):
        '''
        the parameter ignore_cur_state, if set to true indicates that this is a part of a rollback play.

        '''
        agent_state, goal_state, obstacles = self.get_info_from_state(state)

        if agent_state['speed'] == 0 and len(self.agent_state_history) > 0:
            abs_approx_orientation, agent_orientation_index = get_abs_orientation(self.agent_state_history[-1], 
                                                                                self.orientation_approximator)
        else:
            abs_approx_orientation, agent_orientation_index = get_abs_orientation(agent_state, self.orientation_approximator)


        agent_orientation_angle = state['agent_head_dir']
        #print('Current heading direction :', agent_orientation_angle)
        if len(self.agent_state_history) > 0:
            prev_frame_info = self.agent_state_history[-1]
        else:
            prev_frame_info = None

        relative_orientation = get_rel_orientation(prev_frame_info, agent_state, goal_state)
        relative_orientation_goal = get_rel_goal_orientation(self.orientation_approximator,
                                                   self.rel_orient_conv,
                                                   agent_state, 
                                                   agent_orientation_index,
                                                   goal_state)

        change_in_orientation = self.get_change_in_orientation(state['agent_state']['orientation'])


        for i in range(16):
            self.bins[str(i)] = []

        #print('absolute orientation :', abs_approx_orientation.reshape((3,3)))
        #print('relative orientation :', relative_orientation_goal.reshape((3,3)))
        self.populate_orientation_bin(agent_orientation_angle, agent_state, obstacles)

        collision_info = self.compute_bin_info(agent_orientation_angle, agent_state)

        #adding speed information
        speed_info = self.get_speed_info(agent_state)
        
        self.agent_state_history.append(copy.deepcopy(state['agent_state']))

        extracted_feature = np.concatenate((relative_orientation,
                                            relative_orientation_goal,
                                            change_in_orientation,
                                            collision_info.reshape((-1)),
                                            speed_info
                                            ))

        '''
        #***debugging block*****#
        print('Relative orientation :', relative_orientation)
        print('Relative orientation goal :', relative_orientation_goal.reshape(3,3))
        print('Change in orientation :', change_in_orientation)
        pdb.set_trace()
        #****end block****#
        '''
        if self.return_tensor:
            return reset_wrapper(extracted_feature)
        else:
            return extracted_feature



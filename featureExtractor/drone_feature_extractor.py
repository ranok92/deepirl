import sys
import math
import pdb
import itertools
import torch
import numpy as np 
from utils import reset_wrapper, step_wrapper
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
    #the input vectors are [row, col]
    v1_corr = np.array([v1[1], v1[0]])
    v2_corr = np.array([v2[1], v2[0]])
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def get_rot_matrix(theta):
    '''
    returns the rotation matrix given a theta value
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



def get_abs_orientation(agent_state, orientation_approximator):
    #returns the current absolute binned orientation of the agent
    #one of the 8 directions. Dim:8 (this is the default case)
    #for the default case, it additionally returns a 9 dimensional vector
    #if no orientation information is provided it returns 4.

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
            orientation = None
        else:
            orientation = -1
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



def get_rel_orientation(prev_frame_info, agent_state, goal_state):
    #returns the relative orientation of the agent with the direction
    #of the goal. Dim:4
    relative_orientation_vector = np.zeros(3)
    vector_to_goal = goal_state - agent_state['position']
    if prev_frame_info is None:
        agent_orientation = np.array([-1, 0])
    else:
        agent_orientation = agent_state['position'] - prev_frame_info['agent_state']['position']
    diff_in_angle = angle_between(vector_to_goal, agent_orientation)
    #pdb.set_trace()
    if diff_in_angle < np.pi/4:
        rel_orientation = 0

    elif diff_in_angle < np.pi*3/4 and diff_in_angle >= np.pi/4:
        rel_orientation = 1

    else:
        rel_orientation = 2

    relative_orientation_vector[rel_orientation] = 1 
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
        self.prev_frame_info = None
        self.state_rep_size = None

        self.thresh1 = thresh1*step_size
        self.thresh2 = thresh2*step_size


        self.orientation_approximator = [np.array([-2, -2]), np.array([-2,0]),
                                         np.array([-2, 2]), np.array([0, -2]),
                                         np.array([0, 2]), np.array([2, -2]),
                                         np.array([2, 0]), np.array([2,2])]

        self.orientation_approximator_4 = [np.array([-2, 0]), np.array([0, 2]),
                                           np.array([2, 0]), np.array([0, -2])]
        
        self.rel_orient_conv = [7*np.pi/4, 0, 
                                np.pi/4, 6*np.pi/4, 
                                np.pi/2, 5*np.pi/4,
                                np.pi, 3*np.pi/4]
        '''
        self.rel_orient_conv = [np.pi/4, 0, 7*np.pi/4,
                                2*np.pi/4, 6*np.pi/4,
                                3*np.pi/4, 4*np.pi/4, 5*np.pi/4]
        '''
        self.speed_divisions = [0, 0.5, 1, 2, 3, 4]
        self.inner_ring_density_division = [0, 2, 3, 4]
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
        self.hash_variable = None

        self.num_of_speed_blocks = 3
        self.num_of_orient_blocks = 4

        #state rep size = 16*8+9+3
        self.state_rep_size = 162
        self.generate_hash_variable()
        #self.generate_state_dictionary()
        print('Done!')




    def generate_hash_variable(self):
        '''
        The hash variable basically is an array of the size of the current state. 
        This creates an array of the following format:
        [. . .  16 8 4 2 1] and so on.
        '''
        self.hash_variable = np.zeros(self.state_rep_size)
        for i in range(self.hash_variable.shape[0]-1, -1, -1):
    
            self.hash_variable[i] = math.pow(2, self.hash_variable.shape[0]-1-i)

        print(self.hash_variable)


        

    def recover_state_from_hash_value(self, hash_value):

        size = self.state_rep_size
        binary_str = np.binary_repr(hash_value, size)
        state_val = np.zeros(size)
        i = 0
        for digit in binary_str:
            state_val[i] = int(digit) 
            i += 1

        return state_val


    def hash_function(self, state):
        '''
        This function takes in a state and returns an integer which uniquely identifies that 
        particular state in the entire state space.
        '''
        if isinstance(state, torch.Tensor):

            state = state.cpu().numpy()


        return int(np.dot(self.hash_variable, state))



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
        #given an obstacle, the agent state and orientation, 
        #populates the self.bins dictionary with the appropriate obstacles 
        if agent_orientation_val >4:
            agent_orientation_val-=1

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

                if distance < self.thresh1:
                    #classify obstacle in the inner ring
                    ring_1 = True
                else:
                    #classify obstacle in the outer ring
                    ring_2 = True

                #check for the orientation
                #obtain relative orientation
                #get the relative coordinates
                rot_matrix = get_rot_matrix(self.rel_orient_conv[agent_orientation_val])

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
                    angle_diff[i] = angle_between(self.orientation_approximator[i], rel_coord_obs)            
                bin_val = np.argmin(angle_diff)

                if ring_2:
                    bin_val += 8

                #orientation of the obstacle needs to be changed as it will change with the 
                #change in the relative angle. No need to change the speed.
                obs_state['orientation'] = rel_coord_orient_ref - rel_coord_obs
                obs_state['position'] = rel_coord_obs
                self.bins[str(bin_val)].append(obs_state)
        #if the obstacle does not classify to be considered

    def overlay_bins(self, pygame_surface, state):

        #a visualizing tool to debug if the binning is being done properly
        #draws the bins on the game surface for a visual inspection of the 
        #classification of the obstacles in their respective bins
        self.orientation_approximator
        #draw inner ring
        center =  np.array([int(state['agent_state']['position'][1]),
                  int(state['agent_state']['position'][0])])
        pygame.draw.circle(pygame_surface, (0,0,0), center, self.thresh1,2) 
        #draw outer ring 
        pygame.draw.circle(pygame_surface, (0,0,0), center, self.thresh2,2)

        line_start_point = np.array([0, -self.thresh2])
        line_end_point = np.array([0,self.thresh2])
        for i in range(8):
            #draw the lines
            rot_matrix = get_rot_matrix(self.rel_orient_conv[i])
            cur_line_start = np.matmul(rot_matrix, line_start_point) + center
            cur_line_end = np.matmul(rot_matrix, line_end_point) + center
            #pdb.set_trace()
            pygame.draw.line(pygame_surface, (0,0,0), cur_line_start,
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
        #pdb.set_trace()

        return sam_vector, density_inner_ring, density_outer_ring



    def compute_social_force(self):
        #computes the social force value at a given time(optional)

        return 0 



    def extract_features(self, state):
        #getting everything to come together to extract the features

        agent_state, goal_state, obstacles = self.get_info_from_state(state)
        abs_approx_orientation, agent_orientation_index = get_abs_orientation(agent_state, self.orientation_approximator)

        #print('The orientation :')
        #print(abs_approx_orientation.reshape(3,3))

        relative_orientation = get_rel_orientation(self.prev_frame_info, agent_state, goal_state)


        #print('The absolute approx orientation :', abs_approx_orientation)
        ##print('The relative orientation', relative_orientation)

        #empty bins before populating
        for i in range(16):
            self.bins[str(i)] = []


        #print('Here')
        self.populate_orientation_bin(agent_orientation_index, agent_state, obstacles)
        sam_vector, inner_ring_density, outer_ring_density = self.compute_bin_info()

        extracted_feature = np.concatenate((abs_approx_orientation,
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
        self.prev_frame_info = copy.deepcopy(state)

        return reset_wrapper(extracted_feature)



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

            pdb.set_trace()
        
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


        relative_orientation = get_rel_orientation(self.prev_frame_info, agent_state, goal_state)

        for i in range(16):
            self.bins[str(i)] = []

        self.populate_orientation_bin(agent_orientation_index, agent_state, obstacles)

        collision_info = self.compute_bin_info()

        self.prev_frame_info = copy.deepcopy(state)
        #pdb.set_trace()

        #return reset_wrapper(extracted_feature)


        return None






'''
this file contains different types of feature extractors 
specifically for the 10x10 super simplified gridworld environment
'''
import sys
sys.path.insert(0, '..')
import math
import pdb
import itertools
import torch
import numpy as np 
from utils import reset_wrapper, step_wrapper
import os
import copy
'''
    Creating a new class?? Keeps these POINTS in mind.
    
    ***will change to parent and sub class later***
    Each class MUST have the following methods:

        get_info_from_state(self, state)
        
        generate_state_dictionary(self)

        extract_features(self, state)

    Each class MUST have the following members:
    

        self.state_dictionary : A dictionary that contains 
                                all states possible based on the
                                state representation created by
                                the feature extractor with 
                                an index for each of the state.

'''


'''

    THE STATE PUBLISHED BY THE ENVIRONMENT IS A DICTIONARY
    WITH THE FOLLOWING FIELDS so far:
    'agent_state' - numpy array
    'agent_head_dir' - int
    'goal_state' - numpy array
    'obstacles' - list of numpy arrays

'''

#*************array of helper methods***************#

#helper methods





class LocalGlobal():
    #structure of the features first 4 : general direction of the goal
    #                           next 3 : indicates whether the agent moved towards or away from goal
    #                           next n^2 : indicates local obstacle information
    def __init__(self,window_size=5, grid_size = 1, step_size=None,
                agent_width = 1, obs_width = 1):

        self.gl_size = 9
        self.rl_size = 3
        self.window_size = window_size
        if step_size is None:
            self.step_size = grid_size
        else:
            self.step_size = step_size
        self.grid_size = grid_size
        self.agent_width = agent_width
        self.obs_width = obs_width
        self.prev_dist = None
        #added new (26-3-19)
        #based on the state representation, this should contain a 
        #dictionary containing all possible states
        self.state_dictionary = {}
        self.state_str_arr_dict = {}
        self.inv_state_dictionary = {}
        self.hash_variable = None
        self.generate_hash_variable()
        #self.generate_state_dictionary()

    #generates the state dictionary based on the structure of the 
    #hand crafted state space
    
    #the keys in the dictionary are strings converted from 
    #numpy arrays
    

    def generate_hash_variable(self):
        '''
        The hash variable basically is an array of the size of the current state. 
        This creates an array of the following format:
        [. . .  16 8 4 2 1] and so on.
        '''
        self.hash_variable = np.zeros(self.gl_size+self.rl_size+self.window_size**2)
        for i in range(self.hash_variable.shape[0]-1,-1,-1):
    
            self.hash_variable[i] = math.pow(2,self.hash_variable.shape[0]-1-i)

        print(self.hash_variable)

        os.system('pause')

        

    def recover_state_from_hash_value(self, hash_value):

        size = self.gl_size+self.rl_size+self.window_size**2
        binary_str = np.binary_repr(hash_value, size)
        state_val = np.zeros(size)
        i = 0
        for digit in binary_str:
            state_val[i] = int(digit) 
            i += 1

        return state_val


    def hash_function(self,state):
        '''
        This function takes in a state and returns an integer which uniquely identifies that 
        particular state in the entire state space.
        '''
        if isinstance(state, torch.Tensor):

            state = state.cpu().numpy()


        return int(np.dot(self.hash_variable,state))


    def generate_state_dictionary(self):
        
        gl_size = self.gl_size
        rl_size = self.rl_size
        indexval = 0
        for i in range(gl_size):
            for k in range(gl_size,gl_size+rl_size):
                for j in range(0,self.window_size*self.window_size+1):
                    combos = itertools.combinations(range((self.window_size*self.window_size)),j)
                    for combination in combos:
                        state = np.zeros(gl_size+rl_size+self.window_size*self.window_size)
                        '''
                        if i < 4:
                            state[i] = 1
                        else:
                            state[i%4] = 1
                            state[(i+1)%4] = 1
                        '''
                        state[i] = 1
                        state[k] = 1
                        for val in combination:
                            state[gl_size+rl_size+val]=1

                        #the base state
                        #removing the base case, now for the local representation 
                        #only presence of an obstacle will make it 1
                        #state[4+math.floor((self.window_size*self.window_size)/2)] = 1

                        self.state_dictionary[np.array2string(state)] = indexval
                        self.state_str_arr_dict[np.array2string(state)] = state
                        self.inv_state_dictionary[indexval] = state

                        indexval = len(self.state_dictionary.keys())


    #reads the list of fields from the state to create its features
    def get_info_from_state(self,state):

        agent_state = state['agent_state']
        goal_state = state['goal_state']
        obstacles = state['obstacles']

        return agent_state, goal_state, obstacles


    '''
    def determine_index(self,diff_r, diff_c):

        thresh = int((self.agent_width+self.obs_width)/2)
        if abs(diff_r) < thresh and diff_c > thresh: #right
            index = 2
        elif abs(diff_r) < thresh and diff_c < thresh: #left
            index = 6
        elif diff_r > thresh and abs(diff_c) < thresh: #down
            index = 4
        elif diff_r < thresh  and abs(diff_c) < thresh: #up
            index = 0
        elif diff_r > thresh and diff_c > thresh: #quad4
            index = 3
        elif diff_r < 0 and abs(diff_r) > thresh  and diff_c > thresh: #quad1
            index = 1
        elif diff_r < 0 and abs(diff_r) > thresh and diff_c < 0 and abs(diff_c) > thresh: #quad2
            index = 7
        elif diff_r > thresh and diff_c < 0 and abs(diff_c) > thresh: #quad3
            index = 5
        else:
            index = 8

        return index
    '''

    def determine_index(self, diff_r, diff_c):

        thresh = int((self.agent_width+self.grid_size)/2)
        if abs(diff_r) < thresh and diff_c > 0 and abs(diff_c) >= thresh: #right
            index = 5
        elif abs(diff_r) < thresh and diff_c < 0 and abs(diff_c) >= thresh: #left
            index = 3
        elif abs(diff_r) >= thresh and diff_r > 0 and abs(diff_c) < thresh: #down
            index = 7
        elif abs(diff_r) >= thresh  and diff_r < 0 and abs(diff_c) < thresh: #up
            index = 1
        elif abs(diff_r) >= thresh and diff_r > 0 and abs(diff_c) >= thresh and diff_c > 0: #quad4
            index = 8
        elif diff_r < 0 and abs(diff_r) >= thresh  and abs(diff_c) >= thresh and diff_c > 0: #quad1
            index = 2
        elif diff_r < 0 and abs(diff_r) >= thresh and diff_c < 0 and abs(diff_c) >= thresh: #quad2
            index = 0
        elif abs(diff_r) >= thresh and diff_r > 0 and diff_c < 0 and abs(diff_c) >= thresh: #quad3
            index = 6
        else:
            index = 4

        return index

    def closeness_indicator(self,agent_state, goal_state):

        agent_pos = agent_state['position']
        goal_pos = goal_state
        feature = np.zeros(self.rl_size)
        current_dist = np.linalg.norm(agent_pos-goal_pos)

        if self.prev_dist is None or self.prev_dist == current_dist:

            feature[1] = 1
            self.prev_dist = current_dist
            return feature

        if self.prev_dist > current_dist:

            feature[0] = 1

        if self.prev_dist < current_dist:

            feature[2] = 1

        self.prev_dist = current_dist

        return feature


    def block_to_arrpos(self,r,c):
        a = (self.window_size**2-1)/2
        b = self.window_size
        pos = a+(b*r)+c
        return int(pos)

    def extract_features(self,state):

        #pdb.set_trace()
        agent_state, goal_state, obstacles = self.get_info_from_state(state)
        window_size = self.window_size
        block_width = self.grid_size
        step = self.step_size
        window_rows = window_cols = window_size
        row_start =  int((window_rows-1)/2)
        col_start = int((window_cols-1)/2)

        mod_state = np.zeros(12+window_size**2)

        #a = int((window_size**2-1)/2)
        
        agent_pos = agent_state['position']
        goal_pos = goal_state
        diff_r = goal_pos[0] - agent_pos[0]
        diff_c = goal_pos[1] - agent_pos[1]
        '''
        if diff_x >= 0 and diff_y >= 0:
            mod_state[1] = 1
        elif diff_x < 0  and diff_y >= 0:
            mod_state[0] = 1
        elif diff_x < 0 and diff_y < 0:
            mod_state[3] = 1
        else:
            mod_state[2] = 1
        '''
        index = self.determine_index(diff_r, diff_c)
        mod_state[index] = 1

        feat = self.closeness_indicator(agent_state, goal_state)

        mod_state[self.gl_size:self.gl_size+self.rl_size] = feat

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
                    temp_pos = np.asarray([agent_pos[0] + r*step, 
                                agent_pos[1] + c*step])
                    if self.check_overlap(temp_pos,obs_pos):
                        pos = self.block_to_arrpos(r,c)

                        mod_state[pos+self.gl_size+self.rl_size]=1

        return reset_wrapper(mod_state)

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




    def state_to_spatial_representation(self,state):
        '''
        reads the local information of the state and converts that back to a 2d spatial representation
        which would lead to the reading of the given local state representation
        '''
        local_info = state[self.gl_size+self.rl_size:]

        local_info_spatial = np.reshape(local_info,(self.window_size,self.window_size))

        return local_info_spatial






class FrontBackSideSimple():


    def __init__(self, thresh1=1, thresh2=2, thresh3=3, thresh4=4, 
                 agent_width=1, obs_width=1,
                 step_size=1, grid_size=1):

        self.thresh1 = step_size*thresh1
        self.thresh2 = step_size*thresh2
        self.thresh3 = step_size*thresh3
        self.thresh4 = step_size*thresh4

        self.agent_width = agent_width
        self.obs_width = obs_width
        self.step_size = step_size
        self.grid_size = grid_size


        #self.field_list = fieldList
        self.prev_dist = None
        #added new (26-3-19)
        #based on the state representation, this should contain a 
        #dictionary containing all possible states
       
        self.state_dictionary = {}
        self.state_str_arr_dict = {}
        self.inv_state_dictionary = {}
        self.hash_variable = None

        self.state_rep_size = 9+3+16+1

        self.generate_hash_variable()


        print('Loading state space dictionary. . . ')
        #self.generate_state_dictionary()
        print('Done!')



        #generates the state dictionary based on the structure of the 
    #hand crafted state space
    
    #the keys in the dictionary are strings converted from 
    #numpy arrays
    

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




    #generates the state dictionary based on the structure of the 
    #hand crafted state space
    
    #the keys in the dictionary are strings converted from 
    #numpy arrays
    
    def generate_state_dictionary(self):
        indexval = 0
        for i in range(9):
            for j in range(9,12):
                for k in range(0,4*3+1):
                    combos = itertools.combinations(range(4*3),k)
                    for combination in combos:
                        state = np.zeros(24)
                        '''
                        if i < 4:
                            state[i] = 1
                        else:
                            state[i%4] = 1
                            state[(i+1)%4] = 1
                        '''
                        state[i] = 1
                        state[j] = 1
                        for val in combination:
                            state[12+val]=1

                        #the base state
                        #removing the base case, now for the local representation 
                        #only presence of an obstacle will make it 1
                        #state[4+math.floor((self.window_size*self.window_size)/2)] = 1

                        self.state_dictionary[np.array2string(state)] = indexval
                        self.state_str_arr_dict[np.array2string(state)] = state
                        n = np.array2string(
                            np.asarray([0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0, 0, 1.]))

                        if n in self.state_dictionary.keys():
                            print (self.state_dictionary[n])
                        indexval = len(self.state_dictionary.keys())



    #reads the list of fields from the state to create its features
    def get_info_from_state(self,state):

        agent_state = state['agent_state']
        goal_state = state['goal_state']
        obstacles = state['obstacles']

        return agent_state, goal_state, obstacles


    def determine_index(self, diff_r, diff_c):

        thresh = int((self.agent_width+self.grid_size)/2)
        if abs(diff_r) < thresh and diff_c > 0 and abs(diff_c) >= thresh: #right
            index = 5
        elif abs(diff_r) < thresh and diff_c < 0 and abs(diff_c) >= thresh: #left
            index = 3
        elif abs(diff_r) >= thresh and diff_r > 0 and abs(diff_c) < thresh: #down
            index = 7
        elif abs(diff_r) >= thresh  and diff_r < 0 and abs(diff_c) < thresh: #up
            index = 1
        elif abs(diff_r) >= thresh and diff_r > 0 and abs(diff_c) >= thresh and diff_c > 0: #quad4
            index = 8
        elif diff_r < 0 and abs(diff_r) >= thresh  and abs(diff_c) >= thresh and diff_c > 0: #quad1
            index = 2
        elif diff_r < 0 and abs(diff_r) >= thresh and diff_c < 0 and abs(diff_c) >= thresh: #quad2
            index = 0
        elif abs(diff_r) >= thresh and diff_r > 0 and diff_c < 0 and abs(diff_c) >= thresh: #quad3
            index = 6
        else:
            index = 4

        return index



    def closeness_indicator(self, agent_pos, goal_pos):

        feature = np.zeros(3)
        current_dist = np.linalg.norm(agent_pos-goal_pos)

        if self.prev_dist is None or self.prev_dist == current_dist:

            feature[1] = 1
            self.prev_dist = current_dist
            return feature

        if self.prev_dist > current_dist:

            feature[0] = 1

        if self.prev_dist < current_dist:

            feature[2] = 1

        self.prev_dist = current_dist

        return feature

    def get_orientation_distance(self, agent_pos, obs_pos):

        thresh = int((self.agent_width+self.obs_width)/2)
        diff_r = obs_pos[0] - agent_pos[0]
        diff_c = obs_pos[1] - agent_pos[1]
        orient_bin = -1
        dist_bin = -1
        dist = np.linalg.norm(np.array([abs(diff_r),abs(diff_c)]))
        dist = dist - thresh
        is_hit = False
        if dist <= self.thresh4:
            if dist > self.thresh3:
                dist_bin = 3

            elif dist <= self.thresh3 and dist > self.thresh2:
                dist_bin = 2

            elif dist <= self.thresh2 and dist > self.thresh1:
                dist_bin = 1

            else:
                #pdb.set_trace()
                dist_bin = 0 
            #select orientation bin
            if abs(diff_c) >= thresh or abs(diff_r) >= thresh: 
            #atleast one has to be bigger than the thres else it is a collision
                if abs(diff_c) < abs(diff_r):
                    if diff_r > 0: #down
                        orient_bin = 2
                    else: #top
                        orient_bin = 0
                else:
                    if diff_c > 0:#right
                        orient_bin = 1
                    else: #left
                        orient_bin = 3
            else:
                #print('Collision course!!')
                #collision course
                pass
        if dist < 0:
            is_hit = True
                
        return orient_bin, dist_bin, is_hit

    def extract_features(self, state):

        #pdb.set_trace()
        agent_state, goal_state, obstacles = self.get_info_from_state(state)

        mod_state = np.zeros(self.state_rep_size)

        #a = int((window_size**2-1)/2)
        
        agent_pos = agent_state['position']
        goal_pos = goal_state
        diff_r = goal_pos[0] - agent_pos[0]
        diff_c = goal_pos[1] - agent_pos[1]
        '''
        if diff_x >= 0 and diff_y >= 0:
            mod_state[1] = 1
        elif diff_x < 0  and diff_y >= 0:
            mod_state[0] = 1
        elif diff_x < 0 and diff_y < 0:
            mod_state[3] = 1
        else:
            mod_state[2] = 1
        '''
        index = self.determine_index(diff_r,diff_c)
        mod_state[index] = 1
        feat = self.closeness_indicator(agent_pos, goal_pos)

        mod_state[9:12] = feat

        for i in range(len(obstacles)):

            #as of now this just measures the distance from the center of the obstacle
            #this distance has to be measured from the circumferance of the obstacle

            #new method, simulate overlap for each of the neighbouring places
            #for each of the obstacles
            obs_pos = obstacles[i]['position']
            orient, dist, is_hit = self.get_orientation_distance(agent_pos, obs_pos)
            if dist >= 0 and orient >= 0:
                mod_state[12+dist*4+orient] = 1 # clockwise starting from the inner most circle

            if is_hit:
                mod_state[-1]=1
            
        return reset_wrapper(mod_state)




class FrontBackSide():

    def __init__(self,thresh1=1, thresh2=2,
                 thresh3=3, thresh4=4, agent_width=10,
                 obs_width=10, step_size=10,
                 grid_size=1, fieldList=[]):

        #heading direction 0 default top, 1 right ,2 down and 3 left
        self.heading_direction = 0 #no need for previous heading as the
        #coordinates provided as the state are always assumed as top facing
        self.thresh1 = thresh1 * step_size
        self.thresh2 = thresh2 * step_size
        self.thresh3 = thresh3 * step_size
        self.agent_width = agent_width
        self.obs_width = obs_width
        self.step_size = step_size
        self.grid_size = grid_size
        self.field_list = fieldList
        self.prev_dist = None
        self.state_rep_size = 4+9+3+12 #heading, goal_location , proximity indicator, obs_rep
        #the entire table is not needed, only the first row
        #but I am still keeping this if necessary in future
        '''
        format of matrix for coordinate conversion from a direction
        to the other
            to 
        from        top | right |down | left
            top
            right
            down
            left


        '''

        #this is for a [row, col] format in the agent location
        self.rel_pos_transform_table = np.asarray([
                                                [[1, 1], [-1, 1], [-1, -1], [1, -1]],
                                                [[-1, 1], [1, 1], [1, -1], [-1, -1]],
                                                [[-1, -1], [1, -1], [1, 1], [-1, 1]],
                                                [[1, -1], [-1, -1], [-1, 1], [1, 1]]
                                                ])



        self.swap_coord_table = np.asarray([[0,1,0,1],
                                           [1,0,1,0],
                                           [0,1,0,1],
                                           [1,0,1,0]])



        #adding the state dictionary

        self.state_dictionary = {}

        self.generate_state_dictionary()


    #the state space for this should be 2^4*4
    def generate_state_dictionary(self):

        indexval = 0

        for i in range(4):
            
            for j in range(0,5):

                combos = itertools.combinations(range(4),j)

                for combination in combos:

                    state = np.zeros(8)
                    state[i] = 1
                    for val in combination:

                        state[4+val] = 1

                    self.state_dictionary[np.array2string(state)] = indexval
                    indexval+=1




    #reads the list of fields from the state to create its features
    def get_info_from_state(self, state):
        
        #all the spatial information that comes from the environment are of
        #the form (y,x) i.e. (rows,cols)
        state_list = []
        #print('raw state info :', state)
        for field in self.field_list:
            
            if field == 'agent_head_dir':
                
                self.heading_direction = state[field]
            
            else:
                if type(state[field]) is list:
                    for val in state[field]:
                        state_list.append(val)
                else:
                    state_list.append(state[field])

        

        return np.array(state_list)


    #given the current state returns the relative position of 
    #the goal and all of the obstacles
    def get_relative_coords(self, state):
        
        rel_positions = np.zeros((state.shape[0],state.shape[1]))
        agent_pos = state[0]
        for i in range(state.shape[0]):

            #print('Orig :', state[i, :]) 
            rel_positions[i,:] = state[i,:] - agent_pos
            #print('Rel :', rel_positions[i, :])
        return rel_positions

    #update the relative coordinates based on the current heading
    #and the action taken
    def update_relative_coords(self, rel_positions, action):

        #using the action and the current heading, decide the final heading
        #quick reminder 0 - move front 1 - move right 2 - move down 3 - move left
        
        #print('Heading direction :' , self.heading_direction)
        if action == 4:
            multiplying_factor = self.rel_pos_transform_table[0, self.heading_direction]
            swap = self.swap_coord_table[0, self.heading_direction]
        else:
            multiplying_factor = self.rel_pos_transform_table[0, action]
            self.heading_direction = action
            swap = self.swap_coord_table[0, action]
        #print('Rel_position :', rel_positions)
        for i in range(rel_positions.shape[0]):

            
            if swap == 1: 
                #swap the rows and columns
                rel_positions[i, 0], rel_positions[i, 1] = rel_positions[i, 1], rel_positions[i, 0]

            rel_positions[i, 0] = rel_positions[i, 0]*multiplying_factor[0]
            rel_positions[i, 1] = rel_positions[i, 1]*multiplying_factor[1]
        #print('Rel_position updated :', rel_positions)

        return rel_positions



    def determine_index(self, diff_r, diff_c):

        thresh = int((self.agent_width+self.grid_size)/2)
        if abs(diff_r) < thresh and diff_c > 0 and abs(diff_c) >= thresh: #right
            index = 5
        elif abs(diff_r) < thresh and diff_c < 0 and abs(diff_c) >= thresh: #left
            index = 3
        elif abs(diff_r) >= thresh and diff_r > 0 and abs(diff_c) < thresh: #down
            index = 7
        elif abs(diff_r) >= thresh  and diff_r < 0 and abs(diff_c) < thresh: #up
            index = 1
        elif abs(diff_r) >= thresh and diff_r > 0 and abs(diff_c) >= thresh and diff_c > 0: #quad4
            index = 8
        elif diff_r < 0 and abs(diff_r) >= thresh  and abs(diff_c) >= thresh and diff_c > 0: #quad1
            index = 2
        elif diff_r < 0 and abs(diff_r) >= thresh and diff_c < 0 and abs(diff_c) >= thresh: #quad2
            index = 0
        elif abs(diff_r) >= thresh and diff_r > 0 and diff_c < 0 and abs(diff_c) >= thresh: #quad3
            index = 6
        else:
            index = 4

        return index



    def closeness_indicator(self, state_info):
        '''determines if the agent moved closer or farther away from the goal'''
        agent_pos = state_info[0, :]
        goal_pos = state_info[1, :]
        feature = np.zeros(3)
        current_dist = np.linalg.norm(agent_pos-goal_pos)

        if self.prev_dist is None or self.prev_dist == current_dist:

            feature[1] = 1
            self.prev_dist = current_dist
            return feature

        if self.prev_dist > current_dist:

            feature[0] = 1

        if self.prev_dist < current_dist:

            feature[2] = 1

        self.prev_dist = current_dist

        return feature

    def get_orientation_distance(self, agent_pos, obs_pos):
        '''returns the orientation bin and the distance bin of the obstacle
           in comparison to the agent'''
        thresh = int((self.agent_width+self.obs_width)/2)
        diff_r = obs_pos[0] - agent_pos[0]
        diff_c = obs_pos[1] - agent_pos[1]
        orient_bin = -1
        dist_bin = -1
        dist = abs(diff_r)+abs(diff_c)
        dist = dist - thresh - self.step_size
        #print(dist)
        if dist <= self.thresh3:
            if dist >= self.thresh2:
                dist_bin = 2

            elif dist < self.thresh2 and dist >= self.thresh1:
                dist_bin = 1

            else:
                dist_bin = 0 
            #select orientation bin
            if abs(diff_c) > thresh or abs(diff_r) > thresh: 
            #atleast one has to be bigger than the thres else it is a collision
                if abs(diff_c) < abs(diff_r):
                    if diff_r > 0: #down
                        orient_bin = 2
                    else: #top
                        orient_bin = 0
                else:
                    if diff_c > 0:#right
                        orient_bin = 1
                    else: #left
                        orient_bin = 3
            else:
                #print('Collision course!!')
                #collision course
                pass

        #print('dist :',dist_bin)
        #print('orient bin', orient_bin)
        return orient_bin, dist_bin

    def extract_features(self, state):
        '''exctract the features'''
        #pdb.set_trace()
        state = self.get_info_from_state(state)

        rel_positions = self.get_relative_coords(state)
        state = self.update_relative_coords(rel_positions, self.heading_direction)
        #print('Rel state info:', state)
        mod_state = np.zeros(4+9+3+12)

        #a = int((window_size**2-1)/2)
        mod_state[self.heading_direction] = 1
        agent_pos = state[0]
        goal_pos = state[1]
        diff_r = goal_pos[0] - agent_pos[0]
        diff_c = goal_pos[1] - agent_pos[1]
        '''
        if diff_x >= 0 and diff_y >= 0:
            mod_state[1] = 1
        elif diff_x < 0  and diff_y >= 0:
            mod_state[0] = 1
        elif diff_x < 0 and diff_y < 0:
            mod_state[3] = 1
        else:
            mod_state[2] = 1
        '''
        index = self.determine_index(diff_r, diff_c)
        mod_state[4+index] = 1

        feat = self.closeness_indicator(state)

        mod_state[13:16] = feat

        for i in range(2, len(state)):

            #as of now this just measures the distance from the center of the obstacle
            #this distance has to be measured from the circumferance of the obstacle

            #new method, simulate overlap for each of the neighbouring places
            #for each of the obstacles
            obs_pos = state[i]
            orient, dist = self.get_orientation_distance(agent_pos, obs_pos)
            if dist >= 0 and orient >= 0:
                mod_state[16+dist*4+orient] = 1 # clockwise starting from the inner most circle

            
        return reset_wrapper(mod_state)


class OneHot():

    def __init__(self,grid_rows = 10 , grid_cols = 10 ,wrapper =reset_wrapper):

        self.rows = grid_rows
        self.cols = grid_cols
        self.state_size = self.rows*self.cols

        self.state_dictionary = {}
        self.state_str_arr_dict = {}

        self.generate_state_dictionary()


    def generate_state_dictionary(self):

        indexval = 0
        
        for i in range(self.state_size):
            state = np.zeros(self.state_size)
            state[i] = 1
            self.state_dictionary[np.array2string(state)] = indexval
            self.state_str_arr_dict[np.array2string(state)] = state
            indexval+=1

    def get_info_from_state(self,state):

        agent_state = state['agent_state']
        return agent_state



    def extract_features(self,state):

        feature = np.zeros(self.state_size)
        agent_pos = self.get_info_from_state(state)

        index = agent_pos[0]*self.cols+agent_pos[1]
        feature[index] = 1

        feature = reset_wrapper(feature)

        return feature
'''
custom features as of now 3x1, where:

    feat[0] : 1 if agent is moving towards goal
    feat[1] : 1 if agent is neutral
    feat[2] : 1 if agent is moving away from goal

'''
class SocialNav():

    def __init__(self, fieldList = []):

        self.state_dictionary = {}
        self.state_str_arr_dict = {}
        self.field_list = fieldList
        self.prev_dist = None
        self.curr_dist = None

        self.generate_state_dictionary()

    #array of helper Features
    def socialForecesFeatures(self):

        return 0

    def calcDistanceFromGoal(self):

        return 0


    def determine_index(self,diff_x, diff_y):


        if diff_x==0 and diff_y >0: #up
            index = 0
        elif diff_x==0 and diff_y < 0: #down
            index = 2
        elif diff_x > 0 and diff_y == 0: #right
            index = 1
        elif diff_x < 0  and diff_y ==0: #left
            index = 3
        if diff_x > 0 and diff_y > 0: #quad1
            index = 4
        elif diff_x < 0  and diff_y > 0: #quad2
            index = 5
        elif diff_x < 0 and diff_y < 0: #quad3
            index = 6
        elif diff_x >0 and diff_y < 0: #quad4
            index = 7


    def extract_features(self,state):

        feature = np.zeros(11)
        state_info = self.get_info_from_state(state)
        current_dist = np.linalg.norm(state_info[0,:]-state_info[1,:])
        
        agent_pos = state_info[0]
        goal_pos = state_info[1]
        diff_x = goal_pos[0] - agent_pos[0]
        diff_y = goal_pos[1] - agent_pos[1]

        feature[self.determine_index(diff_x, diff_y)] = 1

        if self.prev_dist is None or self.prev_dist == current_dist:

            feature[8+1] = 1
            self.prev_dist = current_dist
            return reset_wrapper(feature)

        if self.prev_dist > current_dist:

            feature[8+0] = 1

        if self.prev_dist < current_dist:

            feature[8+2] = 1

        self.prev_dist = current_dist

        return reset_wrapper(feature)


    def get_info_from_state(self,state):

        state_list = []
        for field in self.field_list:
            if type(state[field]) is list:
                for val in state[field]:
                    state_list.append(val)
            else:
                state_list.append(state[field])


        return np.asarray(state_list)



    def generate_state_dictionary(self):
        counter = 0
        for j in range(8):

            for i in range(8,11):
                state = np.zeros(11)
                state[j] = 1
                state[i] = 1
                self.state_dictionary[np.array2string(state)] = counter
                counter+=1
                self.state_str_arr_dict[np.array2string(state)] = state 








if __name__=='__main__':

    #f = SocialNav(fieldList = ['agent_state','goal_state'])
    #f = LocalGlobal(window_size=7, fieldList=['agent_state', 'goal_state','obstacles'])
    f = FrontBackSide(fieldList = ['agent_state', 'goal_state','obstacles'])
    #print(f.state_dictionary)
    print(f.hash_variable)
    i = f.hash_variable
    print(f.recover_state_from_hash_value(i))
    k = np.zeros(24)

    key = np.array2string(k)
    #print(f.state_str_arr_dict[key])
    #print(f.state_dictionary)



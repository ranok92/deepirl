'''
this file contains different types of feature extractors 
specifically for the 10x10 super simplified gridworld environment
'''
import sys
sys.path.insert(0, '..')

import pdb
import itertools
import numpy as np 
from utils import reset_wrapper, step_wrapper


'''

	THE STATE PUBLISHED BY THE ENVIRONMENT IS A DICTIONARY
	WITH THE FOLLOWING FIELDS so far:
	'agent_state' - numpy array
	'agent_head_dir' - int
	'goal_state' - numpy array
	'obstacles'	- list of numpy arrays

'''

#*************array of helper methods***************#

#helper methods


class LocalGlobal():

	def __init__(self,window_size=5, grid_size = 1,  
				agent_rad = 1, obs_rad = 1 , fieldList = []):

		self.window_size = window_size
		self.grid_size = grid_size
		self.agent_radius = agent_rad
		self.obs_rad = obs_rad
		self.field_list = fieldList

		#added new (26-3-19)
		#based on the state representation, this should contain a 
		#dictionary containing all possible states
		self.state_dictionary = {}


	#generates the state dictionary based on the structure of the 
	#hand crafted state space
	
	#the keys in the dictionary are strings converted from 
	#numpy arrays
	'''
	def generate_state_dictionary(self):

		state = np.zeros(4+self.window_size*self.window_size)

		#the base state
		state[4+floor((self.window_size*self.window_size)/2)] = 1
		for i in range(3):
			for r in range(self.window_size):
				for c in range(self.window_size):
	

	
	this will be a nightmare of a state space of size 4*2^24

	'''






	#reads the list of fields from the state to create its features
	def get_info_from_state(self,state):

		state_list = []
		for field in self.field_list:
			if type(state[field]) is list:
				for val in state[field]:
					state_list.append(val)
			else:
				state_list.append(state[field])

		return np.array(state_list)



	def block_to_arrpos(self,r,c):
		a = (self.window_size**2-1)/2
		b = self.window_size
		pos = a+(b*r)+c
		return int(pos)

	def extract_features(self,state):

		pdb.set_trace()
		state = self.get_info_from_state(state)
		window_size = self.window_size
		block_width = self.grid_size
		window_rows = window_cols = window_size
		row_start =  int((window_rows-1)/2)
		col_start = int((window_cols-1)/2)

		mod_state = np.zeros(4+window_size**2)

		a = int((window_size**2-1)/2)
		mod_state[a+4] = 1
		agent_pos = state[0]
		goal_pos = state[1]
		diff_x = goal_pos[0] - agent_pos[0]
		diff_y = goal_pos[1] - agent_pos[1]

		if diff_x >= 0 and diff_y >= 0:
		    mod_state[1] = 1
		elif diff_x < 0  and diff_y >= 0:
		    mod_state[0] = 1
		elif diff_x < 0 and diff_y < 0:
		    mod_state[3] = 1
		else:
		    mod_state[2] = 1


		for i in range(2,len(state)):

		    #as of now this just measures the distance from the center of the obstacle
		    #this distance has to be measured from the circumferance of the obstacle

		    #new method, simulate overlap for each of the neighbouring places
		    #for each of the obstacles
		    obs_pos = state[i]
		    obs_rad = self.obs_rad
		    for r in range(-row_start,row_start+1,1):
		        for c in range(-col_start,col_start+1,1):
		            #c = x and r = y
		            #pdb.set_trace()
		            temp_pos = np.asarray([agent_pos[0] + r*block_width, 
		            			agent_pos[1] + c*block_width])
		            if np.array_equal(temp_pos,obs_pos):
		                pos = self.block_to_arrpos(r,c)

		                mod_state[pos+4]=1

		return reset_wrapper(mod_state)



class FrontBackSide():

	def __init__(self,window_size=5,grid_size=1,fieldList = []):

		#heading direction 0 default top, 1 right ,2 down and 3 left
		self.heading_direction=0 #no need for previous heading as the
		#coordinates provided as the state are always assumed as top facing
		self.window_size = window_size
		self.sensor_rad = (int)(window_size/2)
		self.grid_size = grid_size
		self.field_list = fieldList
		#the entire table is not needed, only the first row
		#but I am still keeping this if necessary in future
		'''
		format of matrix for coordinate conversion from a direction
		to the other
			to 
		from		top | right |down | left
			top
			right
			down
			left


		'''


		self.rel_pos_transform_table = np.asarray([
												[[1,1],[-1,1],[-1,-1],[1,-1]],
												[[-1,1],[1,1],[1,-1],[-1,-1]],
												[[-1,-1],[1,-1],[1,1],[-1,1]],
												[[1,-1],[-1,-1],[-1,1],[1,1]]
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

		

		for i in range(4):
			
			for j in range(0,5):

				combos = itertools.combinations(range(4),j)

				for combination in combos:

					state = np.zeros(8)
					state[i] = 1
					for val in combination:

						state[4+val] = 1

					self.state_dictionary[np.array2string(state)] = 0




	#reads the list of fields from the state to create its features
	def get_info_from_state(self,state):

		state_list = []
		for field in self.field_list:
			if type(state[field]) is list:
				for val in state[field]:
					state_list.append(val)
			else:
				state_list.append(state[field])

		return np.array(state_list)


	#given the current state returns the relative position of 
	#the goal and all of the obstacles
	def get_relative_coords(self,state):

		rel_positions = np.zeros((state.shape[0],state.shape[1]))
		agent_pos = state[0]
		for i in range(state.shape[0]):
			rel_positions[i,:] = state[i,:]-agent_pos

		return rel_positions

	#update the relative coordinates based on the current heading
	#and the action taken
	def update_relative_coords(self, rel_positions,action):

		#using the action and the current heading, decide the final heading
		#quick reminder 0 - move front 1 - move right 2 - move down 3 - move left
		if action==4:
			multiplying_factor = self.rel_pos_transform_table[0,self.heading_direction]
			swap = self.swap_coord_table[0,self.heading_direction]
		else:
			multiplying_factor = self.rel_pos_transform_table[0,action]
			self.heading_direction = action
			swap = self.swap_coord_table[0,action]

		for i in range(rel_positions.shape[0]):

			if swap==1: 
				#swap the rows and columns
				rel_positions[i,0],rel_positions[i,1] = rel_positions[i,1],rel_positions[i,0]

			rel_positions[i,0] = rel_positions[i,0]*multiplying_factor[0]
			rel_positions[i,1] = rel_positions[i,1]*multiplying_factor[1]
		
		return rel_positions

	def get_goal_pos(self,rel_pos):

		goal_pos = np.zeros(4)
		r = rel_pos[1,0]
		c = rel_pos[1,1]
		if abs(r)>abs(c):
			#front or back
			if r>0:
				#back
				goal_pos[2]=1
			else:
				#front
				goal_pos[0]=1
		if abs(r)<=abs(c):
			#left or right
			if c<0:
				#left
				goal_pos[3]=1
			if c>0:
				#right
				goal_pos[1]=1
		return goal_pos

	#given the correct local_representation returns the state representation
	def rel_coord_to_local_rep(self,rel_pos):
		'''
		convention for local representation same as always:
		0 : front, 1 : right ,2 : back , 3 : left 
		'''
		goal_pos = np.zeros(4)
		local_rep = np.zeros(4)
		window_size = self.window_size
		block_width = self.grid_size
		window_rows = window_cols = window_size
		row_start =  int((window_rows-1)/2)
		col_start = int((window_cols-1)/2)
		#check if they are within the window range and if so do your magic
		#account for the goal
		goal_pos = self.get_goal_pos(rel_pos)

		#account for the obstacles
		for i in range(2,rel_pos.shape[0]):
			#if i==1, this means we are dealing with the goal rather than
			#the obstacles so, that gets added in the goal_pos array
			#rather than the local_rep array, which is just for the obstacles
			if np.all(rel_pos[i,:]>=-self.sensor_rad) and np.all(rel_pos[i,:]<=self.sensor_rad):

				#this value is within range, place it in the right position
				r = rel_pos[i,0]
				c = rel_pos[i,1]
				if abs(r)>abs(c):
					#front or back
					if r>0:
						#back
						local_rep[2]=1

					else:
						#front
						local_rep[0]=1
				if abs(r)<=abs(c):
					#left or right
					if c<0:
						#left
						local_rep[3]=1
					if c>0:
						#right
						local_rep[1]=1

		return np.concatenate((goal_pos,local_rep),axis=0)


	def extract_features(self,state):
		action = state['agent_head_dir']
		state = self.get_info_from_state(state)
		
		if action!=4:
			pass
			#pdb.set_trace()
		rel_coords = self.get_relative_coords(state)
		updated_coords = self.update_relative_coords(rel_coords,action)
		features = self.rel_coord_to_local_rep(updated_coords)

		return reset_wrapper(features)



if __name__=='__main__':

	f = FrontBackSide()
	print(f.state_dictionary)
	print(len(f.state_dictionary.keys()))





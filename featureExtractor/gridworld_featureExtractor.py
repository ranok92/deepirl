'''
this file contains different types of feature extractors 
specifically for the 10x10 super simplified gridworld environment
'''
import pdb
import numpy as np 


class LocalGlobal():

	def __init__(self,window_size=5, grid_size = 1,  
				agent_rad = 1, obs_rad = 1):

		self.window_size = window_size
		self.grid_size = grid_size
		self.agent_radius = agent_rad
		self.obs_rad = obs_rad


	#state is the goal state
	#the state is a list with the following information in the given
	#order : agent_location ,goal_location ,obs_info1 ,obs_info2 .... 
	def extract_features(self,state):

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

		return mod_state


	#helper methods
	def block_to_arrpos(self,r,c):
		a = (self.window_size**2-1)/2
		b = self.window_size
		pos = a+(b*r)+c
		return int(pos)






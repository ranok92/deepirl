#contains the calculation of the state visitation frequency method for the
#gridworld environment

import torch
import numpy as np 

from rlmethods.b_actor_critic import ActorCritic



def createStateAcionTable(policy , rows= 10 , cols=10 , num_actions = 4):
	'''
	given a particular policy and info about the environment on which it is trained
	returns a matrix of size A x S where A is the 
	size of the action space and S is the size of the state space
	things are hard coded here for the gridworld method but you can fiddle 
	with the size of the environment
	'''
	stateActionTable = np.zeros(num_actions, (rows*cols))
	'''
	the states are linearized in the following way row*cols+cols = col 
	of the state visitation freq table 
	'''
	for i in range(rows):
		for j in range(cols):

			state = np.asarray([i,j])
			action = policy(state)
			stateActionTable[:,i*cols+j] = action

	return stateActionTable



def getStateVisitationFreq(policyfile , rows = 10 , cols = 10 , num_actions = 4):

	model = ActorCritic( policy = policyfile)
	policy = model.policy
	stateActionTable = createStateActionTable(policy, rows, cols, num_actions)




if __name__=='__main__':

	getStateVisitationFreq()
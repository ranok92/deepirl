import torch
import numpy as numpy
from envs import gridworld as gw
from rlmethods import actor_critic
world = gw.Gridworld(display=True,rows=10,cols=10, width=10)

for i in range(10):

	s = world.reset()
	done = False
	while not done:

		a = world.takeUserAction()
		print (a)
		nxt,r,done = world.step(a)

		if done:
			break

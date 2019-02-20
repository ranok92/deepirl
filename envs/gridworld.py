import pygame
import numpy as np
import torch
import time

class Gridworld:

	#the numbering starts from 0,0 from topleft corner and goes down and right
	#the obstacles should be a list of 2 dim numpy array stating the position of the 
	#obstacle
	def __init__(self, rows = 10 , cols = 10 , width = 10, goal_state = None, obstacles = None , display = True ,stepReward=0.001):

		#environment information
		pygame.init()
		pygame.key.set_repeat(1,200)
		self.rows = rows
		self.cols = cols
		self.cellWidth = width
		self.upperLimit = np.asarray([self.rows-1, self.cols-1])
		self.lowerLimit = np.asarray([0,0])


		self.agent_state = np.asarray([np.random.randint(0,self.rows-1),np.random.randint(0,self.cols-1)])
		if goal_state==None:
			self.goal_state = np.asarray([np.random.randint(0,self.rows-1),np.random.randint(0,self.cols-1)])
		else:
			self.goal_state = goal_state

		#using manhattan distance
		self.distanceFromgoal = np.sum(np.abs(self.agent_state-self.goal_state))


		self.display = display
		self.gameDisplay = None
		self.gameExit = False

		#some colors for the display
		self.white = (255,255,255)
		self.black = (0,0,0)
		self.green = (0,255,0)
		self.red = (255,0,0)
		self.clock = pygame.time.Clock()

		self.tickSpeed = 30
		self.agent_action_keyboard = [False for i in range(4)]
		#does not matter if none or not.
		self.obstacles = obstacles
		# 0: up, 1: right, 2: down, 3: left
		self.actionArray = [np.asarray([0,-1]),np.asarray([1,0]),np.asarray([0,1]),
							np.asarray([-1,0]),np.asarray([0,0])]
		self.stepReward = 0.1

	def reset(self):

		self.agent_state = np.asarray([np.random.randint(0,self.rows-1),np.random.randint(0,self.cols-1)])
		self.distanceFromgoal = np.sum(np.abs(self.agent_state-self.goal_state))
		self.gameDisplay = pygame.display.set_mode((self.rows*self.cellWidth,self.cols*self.cellWidth))
		pygame.display.set_caption('Your friendly grid environment')

		if self.display:
			self.render()
		return self.agent_state

	#action is a number which points to the index of the action to be taken
	def step(self,action):
		self.clock.tick(self.tickSpeed)
		#print('printing the keypress status',self.agent_action_keyboard)
		self.agent_state = np.maximum(np.minimum(self.agent_state+self.actionArray[action],self.upperLimit),self.lowerLimit)
		reward, done = self.calculateReward() 
		if self.display:
			self.render()

		return self.agent_state, reward, done 

	#the tricky part
	def render(self):

		#render board
		self.clock.tick(self.tickSpeed)

		self.gameDisplay.fill(self.white)

		#render obstacles
		if self.obstacles is not None:
			for obs in self.obstacles:
				pygame.draw.rect(self.gameDisplay, self.red, [obs[0]*self.cellWidth,obs[1]*self.cellWidth,self.cellWidth, self.cellWidth])
			
		#render goal
		pygame.draw.rect(self.gameDisplay, self.green, [self.goal_state[0]*self.cellWidth, self.goal_state[1]*self.cellWidth,self.cellWidth, self.cellWidth])
		#render agent
		pygame.draw.rect(self.gameDisplay, self.black,[self.agent_state[0]*self.cellWidth, self.agent_state[1]*self.cellWidth, self.cellWidth, self.cellWidth])
		pygame.display.update()
		return 0

	#arrow keys for direction
	def takeUserAction(self):
		self.clock.tick(self.tickSpeed)

		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				print("here")
				key = pygame.key.get_pressed()
				if key[pygame.K_UP]:
					return 0
				if key[pygame.K_RIGHT]:
					return 1
				if key[pygame.K_LEFT]:
					return 3
				if key[pygame.K_DOWN]:
					return 2
		return 4


	def calculateReward(self):

		hit = False
		done = False

		if self.obstacles is not None:
			for obs in self.obstacles:
				if np.array_equal(self.agent_state,obs):
					hit = True

		if (hit):
			reward = -1
			done = True

		elif np.array_equal(self.agent_state, self.goal_state):
			reward = 1
			done = True

		else:

			newdist = np.sum(np.abs(self.agent_state-self.goal_state))

			reward = (self.distanceFromgoal - newdist)*self.stepReward

			self.distanceFromgoal = newdist

		return reward, done





if __name__=="__main__":

	world = Gridworld(display=True, obstacles=[np.asarray([1,2])])
	for i in range(100):
		print ("here")
		state = world.reset()
		print (state)
		totalReward = 0
		done = False
		while not done:

			action = world.takeUserAction()
			next_state, reward,done = world.step(action)

			totalReward+=reward
			if done:
				break

			print("reward for the run : ", totalReward)


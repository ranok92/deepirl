import numpy as np
import time
import pdb
import sys
import math
from copy import copy
sys.path.insert(0, '..')


from featureExtractor.gridworld_featureExtractor import LocalGlobal,FrontBackSide
from featureExtractor.drone_feature_extractor import DroneFeatureSAM1


from itertools import count
import utils  # NOQA: E402
from envs.gridworld_clockless import GridWorldClockless

with utils.HiddenPrints():
    import pygame

class GridWorld(GridWorldClockless):

    #the numbering starts from 0,0 from topleft corner and goes down and right
    #the obstacles should be a list of 2 dim numpy array stating the position of the 
    #obstacle
    def __init__(
        self,
        seed = 7,
        rows = 10,
        cols = 10,
        width = 10,
        goal_state = None,
        obstacles = None,
        display = True,
        is_onehot = True,
        is_random = False,
        step_reward=0.001,
        obs_width=None,
        agent_width=None,
        step_size=None,
        step_wrapper=utils.identity_wrapper,
        reset_wrapper=utils.identity_wrapper,
        show_trail = False,
        consider_heading=False,
        buffer_from_obs=0,
        place_goal_manually=False
    ):
        super().__init__(seed = seed,
                       rows = rows,
                       cols = cols,
                       width = width,
                       goal_state = goal_state,
                       obstacles = obstacles,
                       display = display,
                       is_onehot = is_onehot,
                       is_random = is_random,
                       step_reward= step_reward,
                       step_wrapper=step_wrapper,
                       obs_width=obs_width,
                       agent_width=agent_width,
                       step_size=step_size,
                       consider_heading=consider_heading,
                       buffer_from_obs=buffer_from_obs,
                       reset_wrapper=reset_wrapper)


        if display:
            self.clock = pygame.time.Clock()
            self.gameDisplay = None
            self.tickSpeed = 60
            self.show_trail = show_trail
            self.place_goal_manually = place_goal_manually
        self.agent_action_flag = False

        if isinstance(self.obstacles, int):

            num_obs = self.obstacles
            self.obstacles = []
            for i in range(num_obs):
                
                cur_obs = copy(self.default_obs_template)
                cur_obs['id'] = i
                cur_obs['position'] = np.asarray([np.random.randint(self.lower_limit_obstacle[0],self.upper_limit_obstacle[0]),
                                      np.random.randint(self.lower_limit_obstacle[1],self.upper_limit_obstacle[1])])
                
                self.obstacles.append(cur_obs)


        else:

            if obstacles=='By hand':

                self.obstacles = self.draw_obstacles_on_board()


    def draw_obstacles_on_board(self):


        print("printing from here")
     
        self.clock.tick(self.tickSpeed)
        self.gameDisplay = pygame.display.set_mode((self.cols,self.rows))

        self.gameDisplay.fill((255,255,255))

        obstacle_list = []
        RECORD_FLAG = False
        obs_counter = 0
        while True:

            p = pygame.event.get()
            for event in p:

                if event.type== pygame.MOUSEBUTTONDOWN:

                    RECORD_FLAG=True
                if event.type==pygame.MOUSEBUTTONUP:

                    RECORD_FLAG = False

                if event.type==pygame.MOUSEMOTION and RECORD_FLAG:
                    #record the coordinate of the mouse
                    #convert that to a location in the gridworld
                    #store that location into the obstacle_list
                    cur_obs = copy(self.default_obs_template)
                    cur_obs['id'] = obs_counter
                    cur_obs['position'] = ([math.floor(event.pos[1]), math.floor(event.pos[0])])
                    
                    #if grid_loc not in obstacle_list:
                    obstacle_list.append(cur_obs)
                    
                
                #the recording stops when the key 'q' is pressed
                if event.type== pygame.KEYDOWN: #

                    if event.key==113:
                        
                        self.obstacle_list = obstacle_list
                        return obstacle_list

        

        return None
    

    def place_goal(self):

        paused = True
        while paused:
            for event in pygame.event.get():
                if event.type==pygame.MOUSEBUTTONDOWN:
                    (x,y) = pygame.mouse.get_pos()
                    print("ere")
                    self.goal_state[1] = x
                    self.goal_state[0] = y
                    paused = False




    def render(self):

        #render board
        self.gameDisplay = pygame.display.set_mode((self.cols,self.rows))
        self.clock.tick(self.tickSpeed)

        self.gameDisplay.fill(self.white)

        #render obstacles
        if self.obstacles is not None:
            for obs in self.obstacles:
                pygame.draw.rect(self.gameDisplay, self.red, [obs['position'][1]-(self.obs_width/2),obs['position'][0]-(self.obs_width/2),self.obs_width, self.obs_width])
            
        #render goal
        pygame.draw.rect(self.gameDisplay, self.green, [self.goal_state[1]-(self.cellWidth/2), self.goal_state[0]-(self.cellWidth/2),self.cellWidth, self.cellWidth])
        #render agent
        pygame.draw.rect(self.gameDisplay, self.black,[self.agent_state['position'][1]-(self.agent_width/2), self.agent_state['position'][0]-(self.agent_width/2), self.agent_width, self.agent_width])
        if self.show_trail:
            self.draw_trajectory()

        pygame.display.update()
        return 0



    #arrow keys for direction
    def take_user_action(self):
        '''
        takes action from user
        '''
        self.clock.tick(self.tickSpeed)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print('keypreseed')
                pygame.event.wait()
                key = pygame.key.get_pressed()
                if key[pygame.K_UP]:
                    return 0, True
                if key[pygame.K_RIGHT]:
                    return 1, True
                if key[pygame.K_LEFT]:
                    return 3, True
                if key[pygame.K_DOWN]:
                    return 2, True

        return 4, False


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
            norm = np.linalg.norm(sign_arr)
            if norm > 0:
                sign_arr = sign_arr / norm
            def_arr = np.array([1,1])
            action = sign_arr*def_arr
            print('Action taken:',action)
            '''
            if np.hypot(x,y)>_max_agent_speed:
                normalizer = _max_agent_speed/(np.hypot(x,y))
            #print x,y
            else:
                normalizer = 1
            '''
            return self.action_dict[np.array2string(sign_arr*def_arr)], True

        return self.action_dict[np.array2string(np.array([0,0]))], False

    def close_game(self):

        pygame.quit()
#created this to trace the trajectory of the agents whose trajectory informations are provided in the 
#master list


    def draw_arrow(self, base_position , next_position):

        #base_position = (row,col)

        #draw the stalk
        arrow_width  = self.cellWidth*.3 #in pixels
        base_pos_pixel = (base_position+.5)*self.cellWidth
        next_pos_pixel = (next_position+.5)*self.cellWidth
       
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
    

    def reset(self):

        if self.gameDisplay is not None:
            pygame.image.save(self.gameDisplay,'traced_trajectories.png')
        self.pos_history = []

        num_obs=len(self.obstacles)

        #if this flag is true, the position of the obstacles and the goal 
        #change with each reset
        dist_g = self.goal_spawn_clearance
        if self.is_random:

            if not self.freeze_obstacles:
                self.obstacles = []
                for i in range(num_obs):
                    cur_obs = copy(self.default_obs_template)
                    cur_obs['id'] = i 
                    cur_obs['position'] = np.asarray([np.random.randint(self.lower_limit_obstacle[0], self.upper_limit_obstacle[0]),
                                                      np.random.randint(self.lower_limit_obstacle[1], self.upper_limit_obstacle[1])])
                    self.obstacles.append(cur_obs)


            while True:
                flag = False
                self.goal_state = np.asarray([np.random.randint(self.lower_limit_goal[0], self.upper_limit_goal[0]),
                                              np.random.randint(self.lower_limit_goal[1], self.upper_limit_goal[1])])

                for i in range(num_obs):
                    if np.linalg.norm(self.obstacles[i]['position']-self.goal_state) < (self.cellWidth+self.obs_width)/2 * dist_g:

                        flag = True
                if not flag:
                    break

        dist = self.agent_spawn_clearance
        while True:
            flag = False
            self.agent_state['position'] = np.asarray([np.random.randint(self.lower_limit_agent[0],self.upper_limit_agent[0]),
                                           np.random.randint(self.lower_limit_agent[1],self.upper_limit_agent[1])])
            
            for i in range(num_obs):
                if np.linalg.norm(self.obstacles[i]['position']-self.agent_state['position']) < (self.cellWidth+self.agent_width)/2 * dist:
                    flag = True

            if not flag:
                break


        self.distanceFromgoal = np.sum(np.abs(self.agent_state['position']-self.goal_state))
        self.release_control = False
        self.cur_heading_dir = 0
        if self.is_onehot:
            self.state = self.onehotrep()
        else:

            self.state = {}
            self.state['agent_state'] = self.agent_state
            self.state['agent_head_dir'] = 0 #starts heading towards top
            self.state['goal_state'] = self.goal_state

            self.state['release_control'] = self.release_control
            #if self.obstacles is not None:
            self.state['obstacles'] = self.obstacles

        self.pos_history.append(self.agent_state)

 
        pygame.display.set_caption('Your friendly grid environment')
        self.render()

        if self.place_goal_manually:

            self.place_goal()
            
        if self.is_onehot:
            self.state = self.reset_wrapper(self.state)
        return self.state


        #pygame.image.save(self.gameDisplay,'traced_trajectories')


if __name__=="__main__":

    feat_ext = LocalGlobal(window_size=3, agent_width= 10, obs_width=6, grid_size=10,step_size=20)
    #featExt = FrontBackSide(thresh1=1, thresh2=2, thresh3=3,
    #                        fieldList = ['agent_state', 'goal_state', 'obstacles', 'agent_head_dir']) 

    dro_feat = DroneFeatureSAM1()
    world = GridWorld(display=True, is_onehot = False, is_random=True,
                        seed = 0 , obstacles='./real_map.jpg', 
                        step_size=5, buffer_from_obs=0, 
                        rows = 100, cols = 300 , width=30, obs_width=10)

    for i in range(100):
        print ("here")
        state = world.reset()
        state = feat_ext.extract_features(state)
        totalReward = 0
        done = False

        states = []
        states.append(state)
        for i in count(0):
            t = 0
            while t < 1000:

                action,flag = world.take_action_from_user()
                #action = np.random.randint(4)
                print(action)
                #action = 2
                next_state, reward, done, _ = world.step(action)
                if action!=8:
                    
                #print(next_state)
                    state = feat_ext.extract_features(next_state)
                    c = dro_feat.extract_features(next_state)
                #print('The heading :', state[0:4])
                #print('The goal info :', state[4:13].reshape(3, 3))
                #print('THe obstacle infor :', state[16:].reshape(3, 4))
                
                if flag:

                    t += 1
                    #print(world.pos_history)
                    states.append(state)
                if t > 1000 or done:
                    break

            print("reward for the run : ", totalReward)
            print("the states in the traj :", states)
            break


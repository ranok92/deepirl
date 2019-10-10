import numpy as np 
import pdb


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


class PotentialFieldController():

    def __init__(self, k=5, v=1,
                 eta=200,
                 sigma=100, 
                 agent_width=10,
                 obs_width=10,
                 c_limit=5,
                 attr_f_limit=5,
                 rep_f_limit=5,
                 rep_force_dist_limit=60,
                 force_threshold=0.5,
                 consider_heading=True
                 ):

        self.KP = k
        self.KV = v
        self.ETA = eta
        self.SIGMA = sigma
        self.conic_limit = c_limit
        self.rep_force_dist_limit = rep_force_dist_limit
        self.pfield = None
        self.action_array = [np.asarray([-1,0]),np.asarray([-1,1]),
                             np.asarray([0,1]),np.asarray([1,1]),
                             np.asarray([1,0]),np.asarray([1,-1]),
                             np.asarray([0,-1]),np.asarray([-1,-1]),
                             np.asarray([0,0])]



        self.attr_force_lim = attr_f_limit
        self.rep_force_lim = rep_f_limit
        self.globalpath = {}
        self.normalize_force = True
        self.eps = np.finfo(float).eps
        self.force_threshold = force_threshold


    #calculates the attractive force the agent is experiencing given the 
    #position of the agent and the goal
    #returns a 2 dim vector
    def calculate_attractive_force_btwpoints(self,agent_state,goal_state):

        global_coord = agent_state['position']
        #pdb.set_trace()
        goal_vector = goal_state

        if agent_state['orientation'] is None:
            agent_state['orientation'] = np.array([0, 0])

        attr_force =  self.KP*(global_coord - goal_vector) - self.KV*agent_state['orientation']
        if self.normalize_force:

            mag = np.hypot(attr_force[0] , attr_force[1])/self.attr_force_lim
            return  - attr_force/mag
        else:
            return  - attr_force



    #calculates the repulsive force the agent experiences given its position and 
    #the position of the obstacles
    def calculate_repulsive_force_btwpoints(self,agent_state,obs):

        agent = agent_state['position']
        obs = obs['position']
        rho = np.hypot(agent[0]-obs[0], agent[1]-obs[1])+self.eps
        #print ('rho', rho)
        if rho <= self.rep_force_dist_limit:
            force_vector_x = self.ETA*(1.0/rho - 1/self.rep_force_dist_limit)*(1/rho)*(agent[0]-obs[0])
            force_vector_y = self.ETA*(1.0/rho - 1/self.rep_force_dist_limit)*(1/rho)*(agent[1] -obs[1])
            force_mag = np.hypot(force_vector_x , force_vector_y)/self.rep_force_lim

            if self.normalize_force:
                return (force_vector_x/force_mag , force_vector_y/force_mag)
            else:
                return (force_vector_x , force_vector_y)
        else:
            return (0,0)


    #calculates the potential(not being used for now)
    def calculate_positive_potential(self, agent_state, goal_state):
        pforce = None
        dist = np.linalg.norm(agent_state['position']-goal_state)
        if dist < self.conic_limit:

            pforce = 0.5*self.SIGMA*dist**2

        else:

            pforce = (self.conic_limit*self.SIGMA*dist)-(0.5*self.conic_limit**2)

        return pforce

    #calculates the potential (not being used for now)
    def calculate_negative_potential(self, agent_state, obstacle_state):

        pforce = 0

        dist = np.linalg.norm(agent_state['position']-obstacle_state['position'])+self.eps

        if dist <= self.rep_force_dist_limit:

            pforce = 0.5*self.ETA*(1/dist - 1/self.rep_force_dist_limit)**2

        return pforce



    def select_action_from_force(self, force):

        #if force is too small, it doesnt move
        if np.linalg.norm(force) < self.force_threshold:
            return 8

        else:
            similarity_index = np.zeros(8)
            for i in range(8):

                similarity_index[i] = angle_between(force, self.action_array[i])

            #print('Action taken :', np.argmin(similarity_index))
            #print('similarity_index', similarity_index)
            #pdb.set_trace()
            return np.argmin(similarity_index)

    def select_action_play(self, state):
        #given the current state take the apt action
        total_force = 0
        agent_state = state['agent_state']
        goal_state = state['goal_state']
        obstacle_state_list = state['obstacles']

        #calculate force due to goal
        
        f_goal = self.calculate_attractive_force_btwpoints(agent_state, goal_state)
        #print('fgoal - ', f_goal)
        total_force += f_goal
        rep_force = np.array([0.0, 0.0])
        for obs in obstacle_state_list:

            rf = self.calculate_repulsive_force_btwpoints(agent_state, obs)
            rep_force += rf

        total_force += rep_force
        true_action = int(self.select_action_from_force(total_force))
        
        if true_action!=8:
            rel_action = (true_action - state['agent_head_dir'])%8
        else:
            rel_action = true_action
        #pdb.set_trace()
        return rel_action







import numpy as np 
from numba import njit
import pdb

def angle_between(v1, v2):
    v1_conv = v1.astype(np.dtype('float'))
    v2_conv = v2.astype(np.dtype('float'))
    return np.abs(np.arctan2(np.linalg.det(np.stack((v1_conv,v2_conv))), np.dot(v1_conv,v2_conv)))


@njit
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

@njit
def get_norm(vector):

    return np.linalg.norm(vector)

def get_rot_matrix(theta):
    '''
    returns the rotation matrix given a theta value
    '''
    return np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def deg_to_rad(val):
    '''
    converts degree to radians
    '''
    return val*(np.pi)/180


def rad_to_deg(val):
    '''
    converts radians to degrees
    '''
    return val*180/np.pi


class SocialForcesController():

    def __init__(self,
                speed_div,
                orient_div,
                orient_quant,
                field_of_view=np.pi/2,
                desired_speed=1,
                relax_time=2,
                max_force=2
                ):

        self.orientation_array = []
        self.speed_array = []
        self.field_of_view = field_of_view
        self.desired_speed = desired_speed
        self.relaxation_time = relax_time
        self.max_force = max_force
        self.define_action_array(speed_div, orient_div, orient_quant)


    def define_action_array(self, speed_div, orient_div, orient_quant):

        force_lim = self.max_force
        self.speed_array = [i/speed_div*force_lim for i in range(speed_div)]
        orient_range = int((orient_div-1)/2)
        deg_array = [i*orient_quant for i in range(-orient_range, orient_range+1, 1)]
        unit_v = np.array([-1, 0])
        rot_matrix_list = [get_rot_matrix(deg_to_rad(deg)) for deg in deg_array]
        self.orientation_array = [np.matmul(rot_matrix, unit_v) for rot_matrix in rot_matrix_list]
        #self.orientation_array = deg_array
        pdb.set_trace()

    def calculate_attractive_force(self, agent_state, goal_state):
        '''
        calculates the force exerted on the agent because of the
        goal, current position of the goal and the desired speed
        of the agent
            input: agent_state, goal_state
            output: force exerted on the agent (2 dim vector)
        '''
        desired_velocity = goal_state - agent_state['position']
        current_velocity = agent_state['orientation']*agent_state['speed']

        return 1/self.relaxation_time*(desired_velocity-current_velocity)


    def calculate_repulsive_force_btw_points(self, agent_state, obstacle):
        '''
        calculates the repulsive forces acting on the agent due to a single 
        nearby pedestrian(obstacle)
            input: agent_state, obstacle
            output: repulsive force (2 dim vector)
        '''
        agent_pos = agent_state['position']
        obs_pos = obstacle['position']
        diff_pos = agent_pos - obs_pos
        obs_step = obstacle['speed']*obstacle['orientation']

        return 


    def calculate_repulsive_forces(self, agent_state, obstacle_list):
        '''
        calculates the total repulsive force acting on the
        agent at a given time due to pedestrians nearby
            input: agent_state, obstacle_list
            output: repulsive force exerted on the agent (2 dim vector)
        '''
        net_repulsive_force = np.zeros(2)
        for obstacle in obstacle_list:
            net_repulsive_force += self.calculate_repulsive_force_btw_points(agent_state, 
                                                                             obstacle)

        return net_repulsive_force


    def calculate_social_force(self,
                               attractive_forces,
                               repulsive_forces):
        '''
        calculates the net social force being exerted on the agent
            input: different forces acting on the agent
            output: final social forces being acted on the agent (2 dim vector)
        '''

        return attractive_forces+repulsive_forces
    
    def calculate_action_from_force(self, agent_state, cur_heading_dir, force_vector):

        '''
        given the current force being acted on the agent returns the most
        appropriate set of action for the agent
            input: force vector
            output: action integer(containing both change in orientation and speed)

        '''
        rotated_force_vector = np.matmul(get_rot_matrix(deg_to_rad(cur_heading_dir)), 
                                           np.transpose(force_vector))
        orientation_action = np.argmin([np.dot(self.orientation_array[i], 
                                              rotated_force_vector) for i in range(len(self.orientation_array))])
        
        speed_action = 3
        return speed_action*len(self.orientation_array)+orientation_action



    def eval_action(self, state):
        '''
        calculates the action to take given the current state
            input: the current state as seen by the agent
            output: the action taken by the agent in response
                    to the state as stated by social forces
        '''
        agent_state = state['agent_state']
        obstacles = state['obstacles']
        goal_state = state['goal_state']
        cur_heading_dir = state['agent_head_dir']


        attr_force = self.calculate_attractive_force(agent_state, goal_state)
        repr_force = self.calculate_repulsive_forces(agent_state, obstacles)

        net_social_force = self.calculate_social_force(attr_force, repr_force)

        return self.calculate_action_from_force(agent_state, cur_heading_dir, net_social_force)





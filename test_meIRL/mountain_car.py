import gym, sys, time, os
sys.path.insert(0, '..')

import torch
import numpy as np 
import math
import pdb
from utils import reset_wrapper, step_wrapper



class MCFeaturesplain():

    def __init__(self):

        print('They dont do anything')

    def extract_features(self,state):

        return reset_wrapper(state)


class MCFeaturesOnehot():

    def __init__(self, discrete_pos, discrete_vel):

        self.discrete_pos = discrete_pos
        self.discrete_vel = discrete_vel

        self.pos_range = [-1.2, 0.6]
        self.vel_range = [-0.07, 0.07]

        self.state_rep_size = discrete_pos+discrete_vel
        self.generate_hash_variable()

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


    def rescale_value(self, value, discretization_val, val_range):

        rescaled_val = int((value - val_range[0])/(val_range[1]-val_range[0])*discretization_val)
        return rescaled_val

    def extract_features(self, state):

        pos = state[0]
        vel = state[1]

        rescaled_pos = self.rescale_value(pos, self.discrete_pos, self.pos_range)
        pos_vector = np.zeros(self.discrete_pos)
        pos_vector[rescaled_pos] = 1

        rescaled_vel = self.rescale_value(vel, self.discrete_vel, self.vel_range)
        vel_vector = np.zeros(self.discrete_vel)
        vel_vector[rescaled_vel] = 1
        #pdb.set_trace()
        return reset_wrapper(np.concatenate((pos_vector, vel_vector), axis=0))



class MCFeatures():
#3 dimensions for the actions, the actions becomes a part of the features.
#discretize the position in 128 values and each value is provided a binary number
#7+3 = 10 state size
#the same discretization could be applied to the velocity
    def __init__(self, disc_pos, disc_vel):

        self.init = 0
        self.pos_range = [-1.2, 0.6]
        self.vel_range = [-0.07, 0.07]
        self.disc_pos = disc_pos #controls the discretization
        self.disc_vel = disc_vel #controls the discretization of the velocity
        self.state_rep_size = len(np.binary_repr(self.disc_pos))+len(np.binary_repr(self.disc_vel))
        self.generate_hash_variable()

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


    def get_binary_representation(self, value, discretization_val, val_range):
        rep_size = len(np.binary_repr(discretization_val))
        val_representation = np.zeros(rep_size)
        rescaled_val = math.ceil((value - val_range[0])/(val_range[1]-val_range[0])*discretization_val)
        rescaled_val_binary = np.binary_repr(rescaled_val, rep_size)

        for i in range(len(rescaled_val_binary)):
            val_representation[i] = rescaled_val_binary[i]
        return val_representation


    def extract_features(self, state):

        feat = state
        pos = feat[0]
        vel = feat[1]
        pos_repr = self.get_binary_representation(pos, self.disc_pos, self.pos_range)
        vel_repr = self.get_binary_representation(vel, self.disc_vel, self.vel_range)
        return reset_wrapper(np.concatenate((pos_repr,vel_repr),axis=0))



def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0



def extract_features(state):

    feature = state
    return feature

def rollout(env, num_of_trajs):

    feat = MCFeaturesOnehot(10,10)
    global human_agent_action, human_wants_restart, human_sets_pause
    for i in range(num_of_trajs):
        state_list = []
        human_wants_restart = False
        obser = env.reset()
        #pdb.set_trace()
        print(feat.extract_features(obser))
        state_list.append(feat.extract_features(obser))
        skip = 0
        total_reward = 0
        total_timesteps = 0
        done = False
        while not done:
            if not skip:
                #print("taking action {}".format(human_agent_action))
                a = human_agent_action
                total_timesteps += 1
                skip = SKIP_CONTROL
            else:
                skip -= 1

            obser, r, done, info = env.step(a)
            #pdb.set_trace()
            print(np.reshape(feat.extract_features(obser).cpu().numpy(),(2,10)))

            state_list.append(feat.extract_features(obser))

            #print(obser)
            if r != 0:
                print("reward %0.3f" % r)
            total_reward += r
            window_still_open = env.render()
            if window_still_open==False: return False
            if done: break
            if human_wants_restart: break
            while human_sets_pause:
                env.render()
                time.sleep(0.1)
            time.sleep(0.1)
        states_tensor = torch.stack(state_list)
        torch.save(states_tensor,
                   os.path.join(path, 'traj%s.states' % str(i)))
        print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


'''
while 1:
    window_still_open = rollout(env, 10)
    if window_still_open==False: break
'''

if __name__=='__main__':


    env = gym.make('MountainCar-v0')

    path = './exp_traj_mountain_car'
    if os.path.exists('./exp_traj_mountain_car'):
        pass
    else:
        os.mkdir(path)
    human_agent_action = 0
    human_wants_restart = False
    human_sets_pause = False

    ACTIONS = env.action_space.n
    SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                        # ca


    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    print("ACTIONS={}".format(ACTIONS))
    print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
    print("No keys pressed is taking action 0")

    rollout(env, 20)


    #collect_trajectories(env, 10)
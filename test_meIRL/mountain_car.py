import gym, sys, time, os
sys.path.insert(0, '..')
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np 
import math
import pdb
import glob
from irlmethods.deep_maxent import RewardNet

from irlmethods.irlUtils import calculate_expert_svf
from utils import reset_wrapper, step_wrapper

from matplotlib import pyplot as plt
import re
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class MCFeaturesplain():

    def __init__(self):

        print('They dont do anything')

    def extract_features(self,state):

        return reset_wrapper(state)


class MCFeaturesOnehot():

    def __init__(self, disc_pos, disc_vel):

        self.disc_pos = disc_pos
        self.disc_vel = disc_vel

        self.pos_range = [-1.2, 0.6]
        self.vel_range = [-0.07, 0.07]

        self.pos_rep_dim =  disc_pos
        self.vel_rep_dim = disc_vel
        self.state_rep_size = disc_pos+disc_vel
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

        rescaled_pos = self.rescale_value(pos, self.disc_pos, self.pos_range)
        pos_vector = np.zeros(self.disc_pos)
        pos_vector[rescaled_pos] = 1

        rescaled_vel = self.rescale_value(vel, self.disc_vel, self.vel_range)
        vel_vector = np.zeros(self.disc_vel)
        vel_vector[rescaled_vel] = 1
        #pdb.set_trace()
        return reset_wrapper(np.concatenate((pos_vector, vel_vector), axis=0))



class MCFeatures():
#3 dimensions for the actions, the actions becomes a part of the features.
#discretize the position in 128 values and each value is provided a binary number
#7+3 = 10 state size
#the same discretization could be applied to the velocity
    def __init__(self, disc_pos, disc_vel, pad1=False):
        #keeping the pad1 value as 'True' increases the size of the state
        #representation by 1, where in the extra position a '1' is introduced.
        #Mentioned in Fahad et al to introduce stability in the learning process.
        self.init = 0
        self.pos_range = [-1.2, 0.6]
        self.vel_range = [-0.07, 0.07]
        self.disc_pos = disc_pos #controls the discretization
        self.disc_vel = disc_vel #controls the discretization of the velocity\
        self.pad = pad1
        if self.pad:
            self.state_rep_size = len(np.binary_repr(self.disc_pos))+len(np.binary_repr(self.disc_vel))+1
        else:
            self.state_rep_size = len(np.binary_repr(self.disc_pos))+len(np.binary_repr(self.disc_vel))

        self.pos_rep_dim = len(np.binary_repr(self.disc_pos))
        self.vel_rep_dim = len(np.binary_repr(self.disc_vel))
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
        pad = np.ones(1)
        pos_repr = self.get_binary_representation(pos, self.disc_pos, self.pos_range)
        vel_repr = self.get_binary_representation(vel, self.disc_vel, self.vel_range)

        if self.pad:
            return reset_wrapper(np.concatenate((pos_repr,vel_repr,pad),axis=0))
        else:
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


def rollout(env, num_of_trajs):

    feat = MCFeatures(128,8,pad1=False)
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
            print(np.reshape(feat.extract_features(obser).cpu().numpy()[0:8],(1,8)))
            print(feat.extract_features(obser).cpu().numpy()[8:])
            print(feat.extract_features(obser))
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



def plot_true_and_network_reward(reward_network_folder, feature_extractor):

    env = gym.make('MountainCar-v0')
    reward_network_names = glob.glob(os.path.join(reward_network_folder,'*.pt'))

    exhaustive_state_list, true_reward = get_exhaustive_state_list(feature_extractor)
    reward_holder = np.zeros([len(reward_network_names)+1, len(exhaustive_state_list)])

    hidden_dims = [1024, 256]
    net_counter = 0
    for reward_net in sorted(reward_network_names, key=numericalSort):

        reward_net_model = RewardNet(feature_extractor.state_rep_size, hidden_dims)
        print("loading reward_net :", reward_net)
        reward_net_model.load(reward_net)
        reward_net_model.eval()
        reward_net_model.to('cuda')

        for j in range(len(exhaustive_state_list)):

            state = exhaustive_state_list[j]
            reward = reward_net_model(state)
            reward_holder[net_counter][j] = reward

        net_counter+=1

    reward_holder[-1][:] = np.array(true_reward)
    pdb.set_trace()

    
    ##################for visualizing the rewards###############

    conv_arr = np.array([2**i for i in range(7, -1, -1)])
    conv_arr_vel = np.array([2**i for i in range(3, -1, -1)])
    print(conv_arr_vel)
    reward_mat = np.zeros((128,8))
    for i in range(reward_holder.shape[0]-1):
        state_arr = np.zeros(128)
        fig = plt.figure()
        fig2 = plt.figure()
        ax = Axes3D(fig)
        ax2 = fig2.add_subplot(111)
        lx = reward_mat.shape[0]
        ly = reward_mat.shape[1]
        
        xpos = np.arange(0, lx, 1)
        ypos = np.arange(0, ly, 1)

        xpos, ypos = np.meshgrid(xpos, ypos)
        xpos = xpos.flatten()   # Convert positions to 1D array
        ypos = ypos.flatten()
        zpos = np.zeros(lx*ly)

        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        
        #cs = ['r', 'g', 'b', 'y', 'c'] * ly
        for j in range(reward_holder.shape[1]):


            
            state = exhaustive_state_list[j].cpu().numpy()
            pos = state[0:8]
            vel = state[8:]
            print('pos',pos)
            print('vel', vel)
            print(conv_arr)
            reward_mat[int(pos.dot(conv_arr))-1][int(vel.dot(conv_arr_vel)-1)] = reward_holder[i][j]

        ax.bar3d(xpos,ypos,zpos, dx, dy, reward_mat.flatten())
        cax = ax2.matshow(reward_mat, interpolation='nearest')
        fig2.colorbar(cax)
        plt.show()
    #print(reward_holder[:,0:200])
    plt.pcolor(reward_holder)
    #plt.matshow(reward_holder[:,:])
    plt.show()
    return reward_holder

def get_exhaustive_state_list(feature_extractor):

    state_list = []
    true_reward_list = []
    vel_state = feature_extractor.disc_vel
    pos_state = feature_extractor.disc_pos

    vel_incrementor = (feature_extractor.vel_range[1] - feature_extractor.vel_range[0])/vel_state
    pos_incrementor = (feature_extractor.pos_range[1] - feature_extractor.pos_range[0])/pos_state

    ##print(vel_incrementor)
    ##print(pos_incrementor)


    #v = feature_extractor.vel_range[0] + vel_incrementor/2
    p_init = feature_extractor.pos_range[0] + pos_incrementor/2
    for i in range(pos_state):
        p = p_init + i*pos_incrementor
        v_init = feature_extractor.vel_range[0] + vel_incrementor/2
        for j in range(vel_state):
            v = v_init + j*vel_incrementor
            state = np.array([p,v])
            true_reward = p - 0.5 + v*10
            if p > 0.5:
                true_reward_list.append(true_reward)
            else:
                true_reward_list.append(true_reward)
            #print(state)
            state_list.append(feature_extractor.extract_features(state))

    return state_list, true_reward_list


def view_reward_from_trajectory(reward_network_folder, trajectory_folder, feature_extractor):

    '''
    given a reward network folder, a trajectory folder and a feature extractor, this plots the 
    rewards given to each of the trajectories by each of the reward networks in the reward network folder

    '''
    hidden_dims = [1024, 256]
    reward_network_names = glob.glob(os.path.join(reward_network_folder,'*.pt'))
    trajectories = glob.glob(os.path.join(trajectory_folder,'*.states'))
    
    rewards_across_traj_model = np.zeros((len(reward_network_names), len(trajectories)))
    reward_counter = 0
    for reward_net in sorted(reward_network_names, key=numericalSort):

        reward_net_model = RewardNet(feature_extractor.state_rep_size, hidden_dims)
        print("loading reward_net :", reward_net)
        reward_net_model.load(reward_net)
        reward_net_model.eval()
        reward_net_model.to('cuda')
        traj_counter = 0
        for trajectory in trajectories:

            state_list = torch.load(trajectory)
            cur_reward = 0
            for state in state_list:

                reward = reward_net_model(state)
                cur_reward+=reward

            rewards_across_traj_model[reward_counter][traj_counter] = cur_reward
            traj_counter += 1

        reward_counter += 1

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    cax = ax2.matshow(rewards_across_traj_model, interpolation='nearest')
    fig2.colorbar(cax)
    plt.show()
    pdb.set_trace()

    return rewards_across_traj_model

def plot_svf_on_state_space(trajectory_folder, feature_extractor):

    '''
    given a trajectory folder and a feature extractor, plots the 
    svf of the trajectory on the state space as dictated by the feature
    extractor
    '''
    pos_dim = feature_extractor.pos_rep_dim
    vel_dim = feature_extractor.vel_rep_dim
    conv_arr_pos = np.array([2**i for i in range(pos_dim-1, -1, -1)])
    conv_arr_vel = np.array([2**i for i in range(vel_dim-1, -1, -1)])
    trajectories = glob.glob(os.path.join(trajectory_folder,'*.states'))

    for trajectory in trajectories:

        state_list = torch.load(trajectory)
        svf_matrix = np.zeros((feature_extractor.disc_pos, feature_extractor.disc_vel))
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        for state in state_list:

            pos = state[0:pos_dim].cpu().numpy()
            vel = state[pos_dim:].cpu().numpy()
            pos = int(pos.dot(conv_arr_pos))-1
            vel = int(vel.dot(conv_arr_vel))-1

            svf_matrix[pos][vel] += 1

        cax = ax2.matshow(svf_matrix)
        fig2.colorbar(cax)
        plt.show()




'''
def evalualuate_trajectory(trajectory, reward_network):

    #given a trajectory (a sequence of states) and a reward_network
    #this will return the reward obtained by the trajectory
    reward = 0
    for state in trajectory:
        reward += reward_network(state)

    return reward

def mass_trajectory_evaluation(trajectory_folder, reward_network_folder):
'''



'''
while 1:
    window_still_open = rollout(env, 10)
    if window_still_open==False: break
'''

if __name__=='__main__':

    
    '''
    ######## collecting expert svf########
    env = gym.make('MountainCar-v0')

    path = './exp_traj_mountain_car_MCFeatures_128_8_test'
    if os.path.exists(path):
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

    rollout(env, 10)
    #########################################
    '''
    
    ##################getting a better understanding of the rewards
    feature_extractor = MCFeatures(128, 8)
    #print(feature_extractor.state_rep_size)
    #l = plot_true_and_network_reward('/home/abhisek/Study/Robotics/deepirl/test_meIRL/results/Beluga/MountainCar_beluga_MCFeatures_128_8_updated_svf_calc2019-08-26 14:56:46-reg-0-seed-23-lr-0.001/saved-models-rewards/', feature_extractor)
    
    #reward_mat = view_reward_from_trajectory('./results/Beluga/MountainCar_beluga_MCFeatures_128_8_updated_svf_calc2019-08-26 14:56:46-reg-0-seed-23-lr-0.001/saved-models-rewards', 
    #                            './exp_traj_mountain_car_MCFeatures_128_8',
    #                            feature_extractor)
    ##################################################################
    plot_svf_on_state_space('./exp_traj_mountain_car_MCFeatures_128_8', feature_extractor)
    #print(l)
        

    '''
    #### debugging expert svf####
    feature_extractor = MCFeatures(128,8)
    traj_path = './exp_traj_mountain_car_MCFeatures_128_8_test/'
    exp = calculate_expert_svf(traj_path, max_time_steps=3000, feature_extractor=feature_extractor, gamma=1)
    
    state_arr = np.zeros(128)

    conv_arr = np.array([2**i for i in range(7,-1,-1)])
    for key in exp.keys():

        state = feature_extractor.recover_state_from_hash_value(key)
        pos = state[0:8]
        print(pos)
        print(conv_arr)
        state_arr[int(pos.dot(conv_arr))] += exp[key]

    plt.plot(state_arr)
    plt.show()
    sum_svf = 0
    for key in exp.keys():

        sum_svf += exp[key]

    pdb.set_trace()
    '''
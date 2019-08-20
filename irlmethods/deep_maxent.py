'''
Deep maxent as defined by Wulfmeier et. al.
'''

import pdb
import itertools

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections

import numpy as np

import pathlib

import sys
import os
sys.path.insert(0, '..')
import utils  # NOQA: E402

from irlmethods import irlUtils
# from irlmethods.irlUtils import getStateVisitationFreq  # NOQA: E402
from neural_nets.base_network import BaseNN
from torch.nn.utils import clip_grad_norm_

from torch.optim.lr_scheduler import StepLR
from rlmethods.b_actor_critic import Policy

class RewardNet(BaseNN):
    """Reward network"""

    def __init__(self, state_dims, hidden_dims=[128]):
        super(RewardNet, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(state_dims, hidden_dims[0]),
            nn.ReLU(),
        )
        self.hidden_layers = []
        for i in range(1,len(hidden_dims)):
            self.hidden_layers.append(nn.Sequential(
                                                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                                                    nn.ReLU(),
                                                    )
                                      )
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
        )

    def forward(self, x):
        x = self.input(x)
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)

        x = self.head(x)

        return x


'''
***Passing the parameters for the RL block :
    Directly to the RL block in the experiment folder and not through the IRL block as before
'''
'''
    the parameters rlmethod and env take objects of rlmethod and env respectively

'''


class DeepMaxEnt():
    def __init__(
            self,
            traj_path,
            rlmethod=None,
            env=None,
            iterations=10,
            log_intervals=1,
            on_server=True,
            save_folder=None,
            rl_max_episodes=30,
            graft=True,
            hidden_dims=[128],
            regularizer=0.1,
            learning_rate=1e-3,
            scale_svf=True,
            seed=10, 
            clipping_value=None
    ):

        # pass the actual object of the class of RL method of your choice
        self.rl = rlmethod
        self.env = env
        self.max_episodes = iterations
        self.traj_path = traj_path
        self.rl_max_episodes = self.rl.max_ep_length
        self.graft = graft

    # TODO: These functions are replaced in the rl method already, this
        # needs to be made independant somehow
        # self.env.step = utils.step_torch_state()(self.env.step)
        # self.env.reset = utils.reset_torch_state()(self.env.reset)
        '''
        if self.env.is_onehot:
            self.state_size = self.env.reset().shape[0]
        else:
        '''
        self.state_size = self.rl.feature_extractor.extract_features(self.env.reset()).shape[0]
        self.action_size = self.env.action_space.n
        self.reward = RewardNet(self.state_size, hidden_dims)
        self.hidden_dims = hidden_dims

        self.optimizer = optim.Adam(self.reward.parameters(), lr=learning_rate, weight_decay=0.045)
        self.lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)

        self.EPS = np.finfo(np.float32).eps.item()
        self.log_intervals = log_intervals

        self.seed = seed
        self.scale_svf = scale_svf
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.reward = self.reward.to(self.device)

        #making it run on server
        self.on_server = on_server

        self.regularizer = regularizer
        #folders for saving purposes
        self.plot_save_folder = './'+save_folder+'-reg-'+str(self.regularizer)+\
                                '-seed-'+str(self.seed)+'-lr-'+str(learning_rate)+'/plots/'

        self.reward_network_save_folder = './'+save_folder+'-reg-'+\
                                          str(self.regularizer)+'-seed-'+str(self.seed)+'-lr-'+\
                                          str(learning_rate)+'/saved-models-rewards/'
        self.policy_network_save_folder = './'+save_folder+'-reg-'+str(self.regularizer)+'-seed-'+\
                                          str(self.seed)+'-lr-'+str(learning_rate)+'/saved-models/'
    
        if os.path.exists(self.plot_save_folder):
            pass
        else:
            print(self.plot_save_folder)
            os.mkdir(self.plot_save_folder)


        self.clipping = clipping_value

    #******parts being operated on
    def expert_svf(self):
        return irlUtils.expert_svf(self.traj_path, feat=self.rl.feature_extractor)



    def calc_svf_absolute(self, policy, rows=10, cols=10, 
                          actions_space=5, 
                          goalState=np.asarray([0, 0]),
                          episode_length=30):
        return irlUtils.getStateVisitationFreq(policy, rows=rows, cols=cols, 
                                               num_actions=actions_space,
                                               goal_state=goalState,
                                               episode_length=episode_length)


    def agent_svf_sampling(self,num_of_samples = 10000 , env = None,
                            policy_nn = None , reward_nn = None, 
                            episode_length = 20, feature_extractor = None):

        return irlUtils.get_svf_from_sampling(no_of_samples=num_of_samples,
                                            env=  env, policy_nn = policy_nn,
                                            reward_nn = reward_nn ,
                                            episode_length = episode_length,
                                            feature_extractor = feature_extractor)

    #***********


    #***********
    def expert_svf_dict(self,smoothing_window=None):
        
        return irlUtils.calculate_expert_svf(self.traj_path, feature_extractor= self.rl.feature_extractor)
        '''
        return irlUtils.calculate_expert_svf_with_smoothing(self.traj_path, 
                                                            feature_extractor=self.rl.feature_extractor,
                                                            smoothing_window=smoothing_window)
        
        '''
    def agent_svf_sampling_dict(self, num_of_samples=10000 , env=None,
                                policy_nn=None, reward_nn=None, smoothing_window=None, 
                                scale_svf=True, episode_length=20,
                                feature_extractor=None):

         
        return irlUtils.calculate_svf_from_sampling(no_of_samples=num_of_samples,
                                            env=env, policy_nn=policy_nn,
                                            reward_nn=reward_nn, scale_svf=scale_svf,
                                            episode_length=episode_length,
                                            feature_extractor=feature_extractor)
        '''
        return irlUtils.calculate_svf_from_sampling_using_smoothing(no_of_samples=num_of_samples, 
                                                                    env=env, policy_nn=policy_nn, 
                                                                    reward_nn=reward_nn, 
                                                                    episode_length=episode_length,
                                                                    feature_extractor=feature_extractor,
                                                                    window=smoothing_window)

        '''
    #***********






    def calculate_grads(self, optimizer, stateRewards, freq_diff):

        optimizer.zero_grad()
        dot_prod = torch.dot(stateRewards.squeeze(), freq_diff.squeeze())


        #adding L1 regularization
        lambda1 = self.regularizer
        l1_reg = torch.tensor(0,dtype=torch.float).to(self.device)
        grad_mag = torch.tensor(0, dtype=torch.float).to(self.device)

        for param in self.reward.parameters():
            l1_reg += torch.norm(param,1)

        loss = dot_prod+(lambda1*l1_reg) 
        loss.backward()

        #clipping if asked for
        if self.clipping is not None:
            clip_grad_norm_(self.reward.parameters(), self.clipping)


        for param in self.reward.parameters():
            grad_mag +=torch.norm(param.grad, 1)

        #print ('The magnitude of gradients after clipping:', grad_mag)
        #print('The magnitude of the parameters :', l1_reg)


        return loss, dot_prod , l1_reg, grad_mag, torch.norm(stateRewards.squeeze(), 1)


    '''
    def per_state_reward(self, reward_function, rows, cols):
        all_states = itertools.product(range(rows), range(cols))

        oh_states = []
        for state in all_states:
            oh_states.append(utils.to_oh(state[0]*cols+state[1], rows*cols))

        all_states = torch.tensor(oh_states,
                                  dtype=torch.float).to(self.device)

        return reward_function(all_states)

    '''


    def per_state_reward(self, reward_function):

        all_state_list = []
        state_dict = self.rl.feature_extractor.state_str_arr_dict


        for state in state_dict:

            state_tensor = state_dict[state]
       
            all_state_list.append(state_tensor)

        all_states = torch.tensor(all_state_list, dtype=torch.float).to(self.device)

        return reward_function(all_states)


    def get_rewards_of_states(self, reward_function, state_list):


        state_tensors = torch.tensor(state_list, dtype=torch.float).to(self.device)

        return reward_function(state_tensors)


    def plot(self, images, titles, save_path=None):

        nrows = max(1,int(len(images)/2)+1)
        ncols = 2
        colorbars = []

        for image_idx, image in enumerate(images):
            plt.subplot(nrows, ncols, image_idx+1)
            plt.title(titles[image_idx])

            im = plt.imshow(image)
            colorbars.append(plt.colorbar(im))

        plt.pause(0.0001)

        # save the plot
        if save_path:
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

            img_i = 0

            while os.path.exists(os.path.join(save_path, '%s.png' % img_i)):
                img_i += 1

            filename = os.path.join(save_path, '%s.png' % img_i)

            plt.savefig(filename, bbox_inches='tight')

        for cb in colorbars:
            cb.remove()

    def plot_info(self,inp_tuple):
        #pass a tuple containing n number of lists , this function goes through all and plots them
        i = 0
        color_list  = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r']
        for list_val in inp_tuple:
            plt.figure(i)
            plt.plot(list_val,color_list[i])
            plt.draw()
            plt.pause(.0001)
            i += 1
            '''
            plt.figure(1)
            plt.plot(x_axis,svf_diff,'g')
            plt.draw()
            plt.pause(.0001)
            plt.figure(2)
            plt.plot(x_axis,dot_prod,'b')
            plt.draw()
            plt.pause(.0001)
            '''


    def resetTraining(self,inp_size, out_size, hidden_dims, graft=True):

        if graft:
            newNN = Policy(inp_size, out_size, 
                           hidden_dims=hidden_dims,
                           input_net=self.reward.input,
                           hidden_net=self.reward.hidden_layers)
        else:
            newNN = Policy(inp_size, out_size, 
                           hidden_dims=hidden_dims)

        newNN.to(self.device)
        self.rl.policy = newNN
        print(self.rl.policy)
        self.rl.optimizer = optim.Adam(self.rl.policy.parameters(), lr=3e-4)

    #############################################

    def extract_svf_difference(self,svf_dict, svf_array):
        #here the dict is converted to array and the difference is taken
        #diff = array - dict
        svf_diff = []
        svf_from_dict = []
        svf_from_arr = []
        svf_arr2 = np.zeros(len(self.rl.feature_extractor.state_dictionary.keys()))
        for key in svf_dict.keys():
            #print('The key :',key)
            state = self.rl.feature_extractor.recover_state_from_hash_value(key)
            index = self.rl.feature_extractor.state_dictionary[np.array2string(state)]
            svf_arr2[index] = svf_dict[key]

        svf = np.squeeze(svf_array)

        diff = svf_arr2-svf
        print('The sum of all differences :', np.linalg.norm(diff, ord=1))
        '''
        for i in range(diff.shape[0]):

            if diff[i] != 0:
                svf_diff.append(diff[i])  
        '''
        for i in range(svf_array.shape[0]):

            if svf_array[i] != 0 or svf_arr2[i] != 0:
                svf_diff.append(svf_array[i]-svf_arr2[i])

                svf_from_dict.append(svf_arr2[i])
                svf_from_arr.append(svf_array[i])
                
        return svf_diff, svf_from_dict, svf_from_arr

    def relevant_states(self, svf_diff_array):

        state_list = []
        svf_diff_array = np.squeeze(svf_diff_array)
        st_counter = 0
        for i in range(svf_diff_array.shape[0]):

            if svf_diff_array[i] != 0:

                state = self.rl.feature_extractor.inv_state_dictionary[i]
                state_list.append(state)
                st_counter += 1
                print('State :', state , '  ', svf_diff_array[i])

        print('Total states from array :', st_counter)
        return state_list

    def extract_svf_difference_2(self,svf_dict, svf_array):
        #here the array is converted into a dict and then the difference is taken
        svf_array = np.squeeze(svf_array)
        svf_new_dict = {}
        svf_diff_list = []
        for i in range(svf_array.shape[0]):

            if svf_array[i]!=0:
                state = self.rl.feature_extractor.inv_state_dictionary[i]
                print(state)
                hash_value = self.rl.feature_extractor.hash_function(state)
                print(hash_value)
                svf_new_dict[hash_value] = svf_array[i]

        for key in svf_dict.keys():

            if key not in svf_new_dict.keys():

                print('Miss type 1', key )
            else:
                svf_diff_list.append(svf_new_dict[key] - svf_dict[key])

        for key in svf_new_dict.keys():

            if key not in svf_dict.keys():

                print('Miss type 2', key)

        return svf_diff_list


    def array_to_state_dict(self,narray):

        narray = np.squeeze(narray)
        state_dict = {}

        for i in range(narray.shape[0]):

            if narray[i] != 0:
                state = self.rl.feature_extractor.inv_state_dictionary[i]
                hash_value = self.rl.feature_extractor.hash_function(state)

                state_dict[hash_value] = narray[i]

        return collections.OrderedDict(sorted(state_dict.items()))



    #############################################

    def train(self):
        '''
        Contains the code for the main training loop of the irl method.
        Includes calling the RL and environment from within
        '''

        #expertdemo_svf = self.expert_svf()  # get the expert state visitation frequency
        #*****EXPERT SVF COMING FROM POLICY INSTEAD OF 
        #TRAJECTORIES. Will change in future. So handcoded.

        '''
        expert_policy = Policy(self.state_size,self.action_size)
        expert_policy.to(self.device)
        expert_policy.load('./saved-models/g5_5_o1_2.pt')
        expertdemo_svf = self.calc_svf_absolute( expert_policy, 
                                         rows=self.env.rows,
                                         cols=self.env.cols,
                                         goalState = self.env.goal_state,
                                         episode_length = self.rl_max_episodes)

        '''
        #not the best way to call the method but I am too tired to make anything fancy
        #generating svf from samples

        smoothing_window = np.asarray([[0,.1,0],[.1,.6,.1],[0,.1,0]])
        print('Reading expert-svf . . ')
        expertdemo_svf = self.expert_svf_dict(smoothing_window=smoothing_window)
        print('Done reading expert-svf.')
        #expertdemo_svf = self.expert_svf_dict()
        #expert_svf_arr  = self.expert_svf()
        lossList = []
        dot_prod_list = []
        svf_diff_list = []
        l1_reg_list = []
        rewards_norm_list = []

        #added new
        model_performance_list = [] #list to store the raw score obtained by the current policy
        model_performance_nn = []
        reward_grad_norm_list = []

        for i in range(self.max_episodes):
            print('starting iteration %s ...'% str(i))

            # current_agent_policy = self.rl.policy

            self.resetTraining(self.state_size, self.action_size, self.hidden_dims, self.graft)

            #save the reward network

            pathlib.Path(self.reward_network_save_folder).mkdir(parents=True, exist_ok=True)
            self.reward.save(self.reward_network_save_folder)

            #torch.manual_seed(7)
            #np.random.seed(7)
            print('Starting RL training. . .')
            current_agent_policy = self.rl.train_mp(
                n_jobs=4,
                reward_net=self.reward,
                irl=True
            )
            print('Completed RL training.')
            #np.random.seed(11)
            print('Starting sampling agent-svf. . .')
            current_agent_svf, true_reward, nn_reward = self.agent_svf_sampling_dict(num_of_samples=500,
                                                             env=self.env,
                                                             policy_nn=self.rl.policy,
                                                             reward_nn=self.reward,
                                                             feature_extractor=self.rl.feature_extractor,
                                                             episode_length=self.rl_max_episodes,
                                                             smoothing_window=smoothing_window)

            model_performance_list.append(true_reward)
            model_performance_nn.append(nn_reward)

            print('Completed agent-svf sampling.')
            #print('True reward :', true_reward)
            #print('NN reward :', nn_reward)

            #np.random.seed(11)
            #test_agent_svf = self.agent_svf_sampling(num_of_samples=300,
            #                                    env = self.env,
            #                                    policy_nn= self.rl.policy,
            #                                    reward_nn = self.reward,
            #                                    feature_extractor = self.rl.feature_extractor,
            #                                    episode_length = self.rl_max_episodes)
            #save the policy network
            
            #policy_network_folder = './saved-models/'+'loc_glob_win_3_smooth_test_rectified_svf_dict_sub_30-reg'+str(self.regularizer)+'-seed'+str(self.env.seed)+'/'
            pathlib.Path(self.policy_network_save_folder).mkdir(parents=True, exist_ok=True)
            current_agent_policy.save(self.policy_network_save_folder)
            

            states_visited, diff_freq = irlUtils.get_states_and_freq_diff(expertdemo_svf, current_agent_svf, self.rl.feature_extractor)
            svf_diff_list.append(np.linalg.norm(diff_freq,1))

            diff_freq = -torch.from_numpy(np.array(diff_freq)).type(torch.FloatTensor).to(self.device)

            state_rewards = self.get_rewards_of_states(self.reward, states_visited)

            #all_state_rewards = self.per_state_reward(self.reward)

            dot_prod_from_dict = torch.dot(state_rewards.squeeze(), diff_freq.squeeze())

            # GRAD AND BACKPROP

            loss, dot_prod, l1val, reward_nn_grad_magnitude, rewards_norm = self.calculate_grads(self.optimizer, state_rewards,
                                                                        diff_freq)


            lossList.append(loss)
            dot_prod_list.append(dot_prod)
            l1_reg_list.append(l1val)
            rewards_norm_list.append(rewards_norm)
            reward_grad_norm_list.append(reward_nn_grad_magnitude)


            self.plot_info((lossList, svf_diff_list, 
                            l1_reg_list, dot_prod_list, rewards_norm_list, reward_grad_norm_list,
                            model_performance_list, model_performance_nn))

            self.optimizer.step()
            self.lr_scheduler.step()

            print('done')


            #storing the plots in files
            if (i+1) % 3 == 0:

                plt.figure(0)
                file_name = self.plot_save_folder+'loss-iter'+str(i)+'.jpg'
                plt.savefig(file_name)
                
                plt.figure(1)
                file_name = self.plot_save_folder+'svf-diff'+str(i)+'.jpg'
                plt.savefig(file_name)
                
                plt.figure(2)
                file_name = self.plot_save_folder+'l1-reg'+str(i)+'.jpg'
                plt.savefig(file_name)
                
                plt.figure(3)
                file_name = self.plot_save_folder+'dot-prod'+str(i)+'.jpg'
                plt.savefig(file_name)
                
                plt.figure(4)
                file_name = self.plot_save_folder+'rewards-norm'+str(i)+'.jpg'
                plt.savefig(file_name)

                plt.figure(5)
                file_name = self.plot_save_folder+'reward-net-grad-norm'+str(i)+'.jpg'
                plt.savefig(file_name)

                plt.figure(6)
                file_name = self.plot_save_folder+'model-performance-true'+str(i)+'.jpg'
                plt.savefig(file_name)

                plt.figure(7)
                file_name = self.plot_save_folder+'model-performance-nn'+str(i)+'.jpg'
                plt.savefig(file_name)


        return self.reward

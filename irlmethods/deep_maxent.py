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


#adding tensorboard for visualization
from tensorboardX import SummaryWriter

#writer = SummaryWriter('../test_meIRL/tensorboard_log')


class RewardNet(BaseNN):
    """Reward network"""

    def __init__(self, state_dims, hidden_dims=[128]):
        super(RewardNet, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(state_dims, hidden_dims[0]),
            nn.ELU(),
        )
        self.hidden_layers = []
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Sequential(
                                                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                                                    nn.ELU(),
                                                    )
                                      )
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Tanh(),
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
            on_server=True,
            save_folder=None,
            graft=True,
            hidden_dims=[128],
            l1regularizer=0.1,
            learning_rate=1e-3,
            scale_svf=False,
            seed=10,
            rl_max_ep_len=None,
            rl_episodes=None,

            clipping_value=None,
            enumerate_all=False
    ):

        # pass the actual object of the class of RL method of your choice
        self.rl = rlmethod
        self.env = env
        self.max_episodes = iterations
        self.traj_path = traj_path
        if rl_max_ep_len is None:
            self.rl_max_episode_len = self.rl.max_episode_length
        else:
            self.rl_max_episode_len = rl_max_ep_len
        
        if rl_episodes is None:
            self.rl_episodes = self.rl.max_episodes
        else:
            self.rl_episodes = rl_episodes

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
        if self.env.continuous_action:
            self.action_size = self.env.action_space.shape
        else:
            self.action_size = self.env.action_space.n
        self.reward = RewardNet(self.state_size, hidden_dims)

        #############debug###########
        #self.reward.load('/home/abhisek/Study/Robotics/deepirl/test_meIRL/results/Beluga/MountainCar_beluga_MCFeatures_128_8_no_sampling_SVF2019-08-22 08:56:38-reg-0-seed-110-lr-0.0001/saved-models-rewards/3.pt')
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.reward = self.reward.to(self.device)

        self.hidden_dims = hidden_dims

        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(self.reward.parameters(), weight_decay=0, lr=self.learning_rate)

        self.lr_scheduler = StepLR(self.optimizer, step_size=2, gamma=0.9)

        self.EPS = np.finfo(np.float32).eps.item()

        self.seed = seed
        self.scale_svf = scale_svf
 
        self.dtype = torch.float32


        #making it run on server
        self.on_server = on_server

        self.l1regularizer = l1regularizer
        #folders for saving purposes

        self.save_folder_tf = save_folder+'-reg-'+str(self.l1regularizer)+\
                                '-seed-'+str(self.seed)+'-lr-'+str(learning_rate)+'/tf_logs/'

        self.plot_save_folder = save_folder+'-reg-'+str(self.l1regularizer)+\
                                '-seed-'+str(self.seed)+'-lr-'+str(learning_rate)+'/plots/'

        self.reward_network_save_folder = save_folder+'-reg-'+\
                                          str(self.l1regularizer)+'-seed-'+str(self.seed)+'-lr-'+\
                                          str(learning_rate)+'/saved-models-rewards/'

        self.policy_network_save_folder = save_folder+'-reg-'+str(self.l1regularizer)+'-seed-'+\
                                          str(self.seed)+'-lr-'+str(learning_rate)+'/saved-models/'
    
        if os.path.exists(self.plot_save_folder):
            pass
        else:
            #print(self.plot_save_folder)
            os.makedirs(self.plot_save_folder)


        self.clipping = clipping_value
        self.writer = SummaryWriter(self.save_folder_tf)
        self.enumerate_all = enumerate_all
    #******parts being operated on
    ############ array based svf calculation. Not feasible for larger state spaces#######
    #####################################################################################

    '''
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

    '''
    ####################################################################################
    ####################################################################################



    #********************* dictionary based svf calculation***************************
    ##################################################################################
    def expert_svf_dict(self, max_time_steps,
                        feature_extractor, 
                        smoothing=False,
                        gamma=1):
        
        return irlUtils.calculate_expert_svf(self.traj_path, 
                                             max_time_steps=max_time_steps,
                                             feature_extractor=feature_extractor,
                                             smoothing=smoothing,
                                             gamma=gamma)
        '''
        return irlUtils.calculate_expert_svf_with_smoothing(self.traj_path, 
                                                            feature_extractor=self.rl.feature_extractor,
                                                            smoothing_window=smoothing_window)
        
        '''
    def agent_svf_sampling_dict(self, num_of_samples=10000 , env=None,
                                policy_nn=None, reward_nn=None, smoothing=False, 
                                scale_svf=True, episode_length=20, gamma=0.99,
                                feature_extractor=None, enumerate_all=False):

        
        return irlUtils.calculate_svf_from_sampling(no_of_samples=num_of_samples,
                                            env=env, policy_nn=policy_nn,
                                            reward_nn=reward_nn, scale_svf=scale_svf,
                                            episode_length=episode_length,
                                            gamma=gamma,
                                            smoothing=smoothing,
                                            feature_extractor=feature_extractor,
                                            enumerate_all=enumerate_all)
        '''
        return irlUtils.calculate_svf_from_sampling_using_smoothing(no_of_samples=num_of_samples, 
                                                                    env=env, policy_nn=policy_nn, 
                                                                    reward_nn=reward_nn, 
                                                                    episode_length=episode_length,
                                                                    feature_extractor=feature_extractor,
                                                                    window=smoothing_window)

        '''
    #***********
    ####################################################################################
    ####################################################################################





    def calculate_grads(self, stateRewards, freq_diff):
        
        #calculates the gradients on the reward network
        self.optimizer.zero_grad()
        dot_prod = torch.dot(stateRewards.squeeze(), freq_diff.squeeze())

        #adding L1 regularization
        lambda1 = self.l1regularizer
        l1_reg = torch.tensor(0, dtype=torch.float).to(self.device)
        grad_mag = torch.tensor(0, dtype=torch.float).to(self.device)

        for param in self.reward.parameters():
            l1_reg += torch.norm(param, 1)

        #adding back the regularizer term
        loss = dot_prod + lambda1*l1_reg
        

        loss.backward()

        #clipping if asked for
        if self.clipping is not None:
            clip_grad_norm_(self.reward.parameters(), self.clipping)


        for param in self.reward.parameters():
            grad_mag += torch.norm(param.grad, 1)

        #print ('The magnitude of gradients after clipping:', grad_mag)
        #print('The magnitude of the parameters :', l1_reg)
        #pdb.set_trace()
        return loss, dot_prod, l1_reg, grad_mag, torch.norm(stateRewards.squeeze(), 1)


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
        '''
        calculates the rewards of all possible states. Suitable with small state spaces
        '''
        all_state_list = []
        state_dict = self.rl.feature_extractor.state_str_arr_dict


        for state in state_dict:

            state_tensor = state_dict[state]
       
            all_state_list.append(state_tensor)

        all_states = torch.tensor(all_state_list, dtype=torch.float).to(self.device)

        return reward_function(all_states)


    def get_rewards_of_states(self, reward_function, state_list):
        '''
        Calculates the rewards of the states provided in the state_list.
        More in line to handle bigger state spaces
        '''

        state_tensors = torch.tensor(state_list, dtype=torch.float).to(self.device)

        return reward_function(state_tensors)



    ''' Not being used for now.
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
    '''


    def plot_info(self,inp_tuple, name_tuple):
        #pass a tuple containing n number of lists , this function goes through all and plots them
        i = 0
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r']
        for list_val in inp_tuple:
            plt.figure(name_tuple[i])
            plt.plot(list_val, color_list[i])
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

    def save_plot_information(self, iteration, inp_tuple, name_tuple):
        '''
        saves the information provided in the input tuple in their respective files
        '''
        color_list  = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r']
        for i in range(len(inp_tuple)):
            plt.figure(name_tuple[i])
            file_name = self.plot_save_folder+name_tuple[i]+str(iteration)+'.jpg'
            plt.savefig(file_name)

    ''' #reset training is now functions within the RL method
    def resetTraining(self, inp_size, out_size, hidden_dims, graft=True):
        
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
        #print('the rewards of the new policy :')
        #print(self.rl.policy.rewards)
        
        self.rl.optimizer = optim.Adam(self.rl.policy.parameters(), lr=self.rl.lr)
    '''
    #############################################################################
    #############################################################################
    #******************Following are a bunch of debugging methods****************
    '''does not contribute to the main workings of the algorithm '''
    
    def extract_svf_difference(self, svf_dict, svf_array):
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


    def array_to_state_dict(self, narray):

        narray = np.squeeze(narray)
        state_dict = {}

        for i in range(narray.shape[0]):

            if narray[i] != 0:
                state = self.rl.feature_extractor.inv_state_dictionary[i]
                hash_value = self.rl.feature_extractor.hash_function(state)

                state_dict[hash_value] = narray[i]

        return collections.OrderedDict(sorted(state_dict.items()))


    ########################################################################
    ########################################################################



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
        print('Reading expert-svf . . ')
        prev_nn_reward_list = []
        prev_state_list = []
        expertdemo_svf = self.expert_svf_dict(self.rl_max_episode_len,
                                              self.rl.feature_extractor,
                                              smoothing=True,
                                              gamma=1)
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

            self.rl.reset_training()
            #save the reward network

            pathlib.Path(self.reward_network_save_folder).mkdir(parents=True, exist_ok=True)
            self.reward.save(self.reward_network_save_folder)

            #torch.manual_seed(7)
            #np.random.seed(7)
            print('Starting RL training. . .')
            self.rl.train(
                self.rl_episodes,
                self.rl_max_episode_len,
                reward_network=self.reward,
            )

            current_agent_policy = self.rl.policy
            print('Completed RL training.')
            #np.random.seed(11)
            print('Starting sampling agent-svf. . .')
            current_agent_svf, true_reward, nn_reward = self.agent_svf_sampling_dict(num_of_samples=100,
                                                             env=self.env,
                                                             policy_nn=self.rl.policy,
                                                             reward_nn=self.reward,
                                                             gamma=1,
                                                             scale_svf=self.scale_svf,
                                                             feature_extractor=self.rl.feature_extractor,
                                                             episode_length=self.rl_max_episode_len,
                                                             smoothing=True,
                                                             enumerate_all=self.enumerate_all)

            model_performance_list.append(true_reward)
            self.writer.add_scalar('Log_info/model_performance_true', true_reward, i)
            model_performance_nn.append(nn_reward)
            self.writer.add_scalar('Log_info/model_performance_nn', nn_reward, i)
    
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
            
            #diff_freq = expert - current_agent
            states_visited, diff_freq = irlUtils.get_states_and_freq_diff(expertdemo_svf, current_agent_svf, self.rl.feature_extractor)
            
            self.writer.add_scalar('Log_info/svf_difference', np.linalg.norm(diff_freq,1), i)
            svf_diff_list.append(np.linalg.norm(diff_freq,1))
            #print('The norm :',np.sum(np.abs(diff_freq)))
            #print('The norm 2 :', np.linalg.norm(diff_freq,1))
            #print('The norm :',np.sum(np.abs(diff_freq)))

            #now the value is negated, so diff_freq = agent-expert
            diff_freq = -torch.from_numpy(np.array(diff_freq)).type(torch.FloatTensor).to(self.device)


            state_rewards = self.get_rewards_of_states(self.reward, states_visited)


            #all_state_rewards = self.per_state_reward(self.reward)

            #dot_prod_from_dict = torch.dot(state_rewards.squeeze(), diff_freq.squeeze())

            # GRAD AND BACKPROP

            loss, dot_prod, l1val, reward_nn_grad_magnitude, rewards_norm = self.calculate_grads(state_rewards,
                                                                        diff_freq)

            #*************** adding to the tensorboard logger ************
            lossList.append(loss)
            self.writer.add_scalar('Log_info/loss', loss, i)

            dot_prod_list.append(dot_prod)
            self.writer.add_scalar('Log_info/dot_product_val', dot_prod, i)

            l1_reg_list.append(l1val)
            self.writer.add_scalar('Log_info/l1_parameters', l1val, i)

            rewards_norm_list.append(rewards_norm)
            self.writer.add_scalar('Log_info/reward_norm', rewards_norm, i)

            reward_grad_norm_list.append(reward_nn_grad_magnitude)
            self.writer.add_scalar('Log_info/reward_grad_norm', reward_nn_grad_magnitude, i)

            ###############################################################


            self.plot_info((lossList, svf_diff_list, 
                            l1_reg_list, dot_prod_list, rewards_norm_list, 
                            reward_grad_norm_list, model_performance_list, 
                            model_performance_nn),
                            ('loss-iter', 'svf-diff', 'l1-reg', 
                            'dot-prod', 'reward-norm',
                            'reward-net-grad-norm', 'model-performance-true',
                            'model-performance-nn')
                            )



            #pdb.set_trace()
            self.optimizer.step()

            self.lr_scheduler.step()
            #print("The current learning rate after itertaion : {} is {}".format(i, 
            #                                                self.lr_scheduler.get_lr()))

            print('done')

            if len(prev_state_list) > 0 and len(prev_nn_reward_list) >0:
                
                cur_reward_list = self.get_rewards_of_states(self.reward, prev_state_list)
                irlUtils.save_bar_plot(prev_nn_reward_list,cur_reward_list, 
                                   prev_diff, i, self.plot_save_folder)


            prev_state_list = states_visited
            prev_nn_reward_list = state_rewards
            prev_diff = diff_freq
            #storing the plots in files
            '''
            no need for storing files as the plots are being monitored
            by tensorboard
            if (i+1) % 3 == 0:

                self.save_plot_information(i,
                            (lossList, svf_diff_list, 
                            l1_reg_list, dot_prod_list, rewards_norm_list, 
                            reward_grad_norm_list, model_performance_list, 
                            model_performance_nn),
                            ('loss-iter', 'svf-diff', 'l1-reg', 'dot-prod', 'reward-norm',
                              'reward-net-grad-norm', 'model-performance-true',
                              'model-performance-nn')
                            )
            '''
        self.writer.close()
        return self.reward

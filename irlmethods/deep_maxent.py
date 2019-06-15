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

import numpy as np

import pathlib

import sys
import os
sys.path.insert(0, '..')
import utils  # NOQA: E402
from irlmethods import irlUtils
# from irlmethods.irlUtils import getStateVisitationFreq  # NOQA: E402
from neural_nets.base_network import BaseNN

from rlmethods.b_actor_critic import Policy


class RewardNet(BaseNN):
    """Reward network"""

    def __init__(self, state_dims):
        super(RewardNet, self).__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.body(x)
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
            on_server = True,
            plot_save_folder=None,
            rl_max_episodes = 30,
            graft = True,
            regularizer=0.1
    ):

        # pass the actual object of the class of RL method of your choice
        self.rl = rlmethod
        self.env = env
        self.max_episodes = iterations
        self.traj_path = traj_path
        self.rl_max_episodes = rl_max_episodes
        self.graft = graft

        self.plot_save_folder = plot_save_folder

        # TODO: These functions are replaced in the rl method already, this
        # needs to be made independant somehow
        # self.env.step = utils.step_torch_state()(self.env.step)
        # self.env.reset = utils.reset_torch_state()(self.env.reset)
        if self.env.is_onehot:
            self.state_size = self.env.reset().shape[0]
        else:
            self.state_size = self.rl.feature_extractor.extract_features(self.env.reset()).shape[0]
        self.action_size = self.env.action_space.n
        self.reward = RewardNet(self.state_size)
        self.optimizer = optim.Adam(self.reward.parameters(), lr=1e-2)
        self.EPS = np.finfo(np.float32).eps.item()
        self.log_intervals = log_intervals

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.reward = self.reward.to(self.device)

        #making it run on server
        self.on_server = on_server

        self.regularizer = regularizer



    #******parts being operated on
    def expert_svf(self):
        return irlUtils.expert_svf(self.traj_path,state_dict= self.rl.feature_extractor.state_dictionary)


    def calc_svf_absolute(self, policy, rows = 10, cols = 10, 
                    actions_space=5 , 
                    goalState= np.asarray([0,0]),
                    episode_length = 30):
        return irlUtils.getStateVisitationFreq(policy, rows=rows, cols=cols, 
                                                num_actions=actions_space ,
                                                goal_state=goalState ,
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
    def expert_svf_dict(self):
        return irlUtils.calculate_expert_svf(self.traj_path, feature_extractor= self.rl.feature_extractor)



    def agent_svf_sampling_dict(self,num_of_samples = 10000 , env = None,
                            policy_nn = None , reward_nn = None,
                            episode_length = 20, feature_extractor = None):

        return irlUtils.calculate_svf_from_sampling(no_of_samples=num_of_samples,
                                            env=  env, policy_nn = policy_nn,
                                            reward_nn = reward_nn ,
                                            episode_length = episode_length,
                                            feature_extractor = feature_extractor)

    #***********






    def calculate_grads(self, optimizer, stateRewards, freq_diff):
        optimizer.zero_grad()
        dotProd = torch.dot(stateRewards.squeeze(), freq_diff.squeeze())
        
        #adding L1 regularization
        lambda1 = self.regularizer
        l1_reg = torch.tensor(0,dtype=torch.float).to(self.device)
        for param in self.reward.parameters():
            l1_reg += torch.norm(param,1)

        loss = dotProd+(lambda1*l1_reg)   
        loss.backward()
        return loss, dotProd,
               (lambda1*l1_reg), torch.norm(stateRewards.squeeze(), 1)


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

    def plot_info(self,inp_list):

        color_list = ['r','g','b','c','m']
        i = 0
        for val in inp_list:
            plt.figure(i)
            plt.plot(val, color_list)
            plt.draw()
            plt.pause(0.0001)
            i += 1


    def resetTraining(self,inp_size,out_size, graft=True):

        if graft:
            newNN = Policy(inp_size,out_size, body_net = self.reward.body)
        else:
            newNN = Policy(inp_size,out_size)

        newNN.to(self.device)
        self.rl.policy = newNN
        self.rl.optimizer = optim.Adam(self.rl.policy.parameters(), lr=3e-4)


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

        expertdemo_svf = self.expert_svf_dict()
        lossList = []
        dot_prod_list = []
        svf_diff_list = []
        l1_reg_list = []
        rewards_norm_list = []

        for i in range(self.max_episodes):
            print('starting iteration %s ...'% str(i))

            # current_agent_policy = self.rl.policy

            self.resetTraining(self.state_size,self.action_size, self.graft)

            #save the reward network
            reward_network_folder = './saved-models-rewards/'+'loc_glob_simple_rectified--'+str(self.regularizer)+'/'

            pathlib.Path(reward_network_folder).mkdir(parents=True, exist_ok=True)
            self.reward.save(reward_network_folder)

            torch.manual_seed(7)
            np.random.seed(7)
            current_agent_policy = self.rl.train_mp(
                n_jobs=1,
                reward_net=self.reward,
                irl=True
            )
            current_agent_svf = self.agent_svf_sampling_dict(num_of_samples = 300,
                                                env = self.env,
                                                policy_nn= self.rl.policy,
                                                reward_nn = self.reward,
                                                feature_extractor = self.rl.feature_extractor,
                                                episode_length = self.rl_max_episodes)


            #save the policy network
            policy_network_folder = './saved-models/'+'loc_glob_simple_rectified--'+str(self.regularizer)+'/'
            pathlib.Path(policy_network_folder).mkdir(parents=True, exist_ok=True)
            current_agent_policy.save(policy_network_folder)

            diff = np.squeeze(expertdemo_svf - current_agent_svf)
            svf_diff_list.append(np.dot(diff,diff))
            diff_freq = -torch.from_numpy(expertdemo_svf - current_agent_svf).type(self.dtype)
            diff_freq = diff_freq.to(self.device)

            #***********changing this block
            #diff_freq = -torch.from_numpy(expertdemo_svf - current_agent_svf).type(self.dtype)
            #diff_freq = diff_freq.to(self.device)

            # returns a tensor of size (no_of_states x 1)
            #reward_per_state = self.per_state_reward(
            #    self.reward)
            #*******************************

            states_visited, diff_freq = irlUtils.get_states_and_freq_diff(expertdemo_svf, current_agent_svf, self.rl.feature_extractor)
            diff_freq_list.append(np.dot(diff_freq,diff_freq))
            diff_freq = torch.from_numpy(np.array(diff_freq)).type(torch.FloatTensor).to(self.device)

            state_rewards = self.get_rewards_of_states(self.reward, states_visited)

            # PLOT
            '''
            to_plot = []
            to_plot.append(diff_freq.cpu().numpy().reshape((10,10)))
            to_plot.append(expertdemo_svf.reshape((10,10)))
            to_plot.append(current_agent_svf.reshape((10,10)))
            to_plot.append(reward_per_state.cpu().detach().numpy().reshape((10,10)))

            to_plot_descriptions = []
            to_plot_descriptions.append('SVF difference (L)')
            to_plot_descriptions.append('expert SVF')
            to_plot_descriptions.append('policy SVF')
            to_plot_descriptions.append('Reward per state')

            self.plot(to_plot, to_plot_descriptions,
                      save_path=self.plot_save_folder)
            '''
            # GRAD AND BACKPROP
            loss, dot_prod, l1val , rewards_norm = self.calculate_grads(self.optimizer, reward_per_state, diff_freq)

            lossList.append(loss)
            dot_prod_list.append(dot_prod)
            l1_reg_list.append(l1val)
            rewards_norm_list.append(rewards_norm)
            
            self.plot_info((lossList, svf_diff_list, 
                            l1_reg_list, dot_prod_list, rewards_norm_list))
            self.optimizer.step()

            print('done')

            if i+1 % 3 == 0:

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


        return self.reward

'''An environment independant actor critic method.'''
import argparse
import pdb
import os
import pathlib
import datetime
import copy

from itertools import count
from collections import namedtuple
import gym
import numpy as np

import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
import pygame
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from gym_envs import np_frozenlake  # NOQA: E402
import utils  # NOQA: E402
from neural_nets.base_network import BasePolicy
#from rlmethods.rlutils import LossBasedTermination

import gc
import psutil


process = psutil.Process(os.getpid())
def display_memory_usage(memory_in_bytes):

    units = ['B', 'KB', 'MB', 'GB', 'TB']
    mem_list = []
    cur_mem = memory_in_bytes
    while cur_mem > 1024:

        mem_list.append(cur_mem%1024)
        cur_mem /= 1024
    
    mem_list.append(cur_mem)
    for i in range(len(mem_list)):

        print(units[i] +':'+ str(mem_list[i])+', ', end='')

    print('\n')



parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


class Policy(BasePolicy):
    """Policy network"""

    def __init__(self, state_dims, action_dims, hidden_dims=[128], 
                 input_net=None, hidden_net=None):
        super(Policy, self).__init__()
        self.hidden_layers = []
        if input_net or hidden_net:
            self.graft(input_net, hidden_net)
        else:
            self.input = nn.Sequential(
                nn.Linear(state_dims, hidden_dims[0]),
                nn.ELU()
            )
            for i in range(1, len(hidden_dims)):
                self.hidden_layers.append(nn.Sequential(nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                                                    nn.ELU()
                                                    )
                                          )

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.action_head = nn.Linear(hidden_dims[-1], action_dims)
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        self.saved_actions = []
        self.saved_entropy = []
        self.rewards = []

    def forward(self, x):
        x = self.input(x)

        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)

        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


    def graft(self, input_net, hidden_net):
        """Grafts a deep copy of another neural network's body into this
        network. Requires optimizer to be reset after this operation is
        performed.

        :param body: body of the neural network you want grafted.
        """
        assert input_net is not None, 'NN body being grafted is None!'

        self.input = copy.deepcopy(input_net)

        assert hidden_net is not None, 'No hidden layers to graft!'
        self.hidden_layers = []
        for i in range(len(hidden_net)):

            self.hidden_layers.append(copy.deepcopy(hidden_net[i]))

        self.hidden_layers = nn.ModuleList(self.hidden_layers)



    def sample_action(self, state):
        """based on current policy, given the current state, select an action
        from the action space.

        :param state: Current state in environment.
        """
        probs, state_value = self.__call__(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action),
                                                     state_value))

        #calculating the entropy term for the policy
        entropy = -torch.sum(probs.mean() * torch.log(probs))
        self.saved_entropy.append(entropy)

        return action.item()


    def eval_action(self, state):
        '''
        use this function to play, as the other one keeps storing information which is not needed
        when evaluating.
        '''

        probs, _ = self.__call__(state)
        _, ind = torch.max(probs, 0)

        return ind.item()



class PolicyRegression(BasePolicy):

    def __init__(self, input_layer_dims, hidden_layer_dims,
                 input_net=None,
                 hidden_net=None):
        """
        Initialize a network given the details of the layers and number of nodes per layer.
        """
        super(PolicyRegression, self).__init__()
        
        self.hidden_layers = []

        self.input_layer = nn.Sequential(
            nn.Linear(input_layer_dims, hidden_layer_dims[0]),
            nn.ELU()
        )
        for i in range(1, len(hidden_layer_dims)):
            self.hidden_layers.append(nn.Sequential(nn.Linear(hidden_layer_dims[i-1], hidden_layer_dims[i]),
                                                nn.ELU()
                                                )
                                        )

        self.hidden_layer = nn.ModuleList(self.hidden_layers)
        
        self.orientation_head = nn.Sequential(
                                            nn.Linear(hidden_layer_dims[-1], hidden_layer_dims[-1]),
                                            nn.Sigmoid(),
                                            nn.Linear(hidden_layer_dims[-1], 1)
                                            )
                                              
        self.speed_head = nn.Sequential(
                                        nn.Linear(hidden_layer_dims[-1], hidden_layer_dims[-1]),
                                        nn.Sigmoid(),
                                        nn.Linear(hidden_layer_dims[-1], 1)
                                         )
        
        self.value_head = nn.Linear(hidden_layer_dims[-1], 1)
        
        #lists for bookkeeping
        self.saved_actions = []
        self.saved_entropy = []
        self.rewards = []


    def forward(self, x):
        """
        Forward propagation through the network
        """
        x = self.input_layer(x)
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers(x)
        
        orient_vals = self.orientation_head(x)
        speed_vals = self.speed_head(x)
        state_values = self.value_head(x)

        return orient_vals, speed_vals, state_values

    def sample_action(self, x):
        action = self.eval_action(x)


        return action
    
    def eval_action(self, x):

        return 0

    def eval_action_continuous(self, x):
        return 0


class ActorCritic:
    """Actor-Critic method of reinforcement learning."""

    def __init__(self, env, feat_extractor= None, policy=None, 
                 termination=None, gamma=0.99, render=False,
                 log_interval=100, max_episodes=0, max_episode_length=200, 
                 hidden_dims=[128], lr=0.001,
                 reward_threshold_ratio=0.99, plot_loss=False, 
                 save_folder=None, entropy_coeff=0):
        """__init__

        :param env: environment to act in. Uses the same interface as gym
        environments.
        """

        self.gamma = gamma
        self.render = render
        self.log_interval = log_interval
        self.max_episodes = max_episodes
        self.max_episode_length = max_episode_length
        self.reward_threshold_ratio = reward_threshold_ratio

        self.env = env
        self.feature_extractor = feat_extractor

        self.termination = termination

        '''
        if env.is_onehot:
            state_size = env.reset().shape[0]
        else:
        '''
        state_size = self.feature_extractor.extract_features(env.reset()).shape[0]

        print("Actor Critic initialized with state size ", state_size)
        # initialize a policy if none is passed.
        self.hidden_dims = hidden_dims
        if policy is None:
            self.policy = Policy(state_size, env.action_space.n, self.hidden_dims)
        else:
            self.policy = policy

        # use gpu if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 'cpu')

        self.policy = self.policy.to(self.device)

        # optimizer setup
        self.lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.EPS = np.finfo(np.float32).eps.item()
        self.entropy_coeff = entropy_coeff
        #for plotting loss
        self.plot_loss = plot_loss
        #stores two things, the plot for loss at the end of each RL iteration 
        #the plot for the reward obtained throughout the training from each of the threads

        if save_folder is not None:
            self.save_folder = save_folder+'/RL_Training'
        else:
            self.save_folder = None
        
        self.loss_interval = min(10, self.log_interval)
        self.loss_mean = []
        self.loss = []



    def generate_trajectory_user(self, num_trajs , path):

        for traj_i in range(num_trajs):

            actions = []
            if self.feature_extractor is None:
                states = [self.env.reset()]
            else:
                states = [self.feature_extractor.extract_features(self.env.reset())]

            done = False
            for i in count(0):
                t= 0
                while t < self.max_episode_length:
                
                    action,action_flag = self.env.take_action_from_user()

                    state, rewards, done, _ = self.env.step(action)
                    
                    if self.feature_extractor is not None:
                        state = self.feature_extractor.extract_features(state)

                    if action_flag:
                        t+=1
                        print("current state :", state)
                        states.append(state)
                        actions.append(action)
                    if t >= self.max_episode_length or done:
                        break

                print ("the states traversed : ", states)

                break
                
            actions_tensor = torch.tensor(actions)
            states_tensor = torch.stack(states)

            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            torch.save(actions_tensor,
                       os.path.join(path, 'traj%s.acts' % str(traj_i)))

            torch.save(states_tensor,
                       os.path.join(path, 'traj%s.states' % str(traj_i)))




    def generate_trajectory(self, num_trajs, render, store_raw=False, 
                            path=None, expert_svf=None,
                            capture_frame_interval=None):
        """
        Generates trajectory based on the requirements.

        input:
            num_trajs : int containing the number of trajectories to generate.
            render : Boolean (Flag to determine if rendering is needed)
            store_raw : Boolean (Flag to determine if the states to save are state
                        dictionaries.)
            path : string, that points to the directory to save the trajectories

            expert_svf : an svf dictionary of the following format 
                            (key : value) - (state_hash_function : freqency)

            capture_frame_interval : int/None denotes the intervals after 
                                     which a screenshot of the game environment 
                                     will be taken (Primairy for capturing sequences
                                     of motion)


        output:

            reward_across_trajs : list denoting the reward obtained by the agent at
                                  each of the runs.

            frac_unknown_states_enc : float denoting the fraction of states encountered
                                      by the current agent that were novel (not encountered
                                      by the expert)

            subject_list : list of the pedestrian ids in the order they were played

            captured_framList : a 2D list where each 1d list is a sequence of screen capture 
                                image arrays from a given trajectory.
                                The order of the trajectories stored in the 2d list is the same 
                                as the order of the pedestrians in the subject list

        """
        reward_across_trajs = []
        frac_unknown_states_enc = []
        master_captured_frame_list = []
        subject_list = None

        if self.env.replace_subject:
            subject_list = []
        

        for traj_i in range(num_trajs):
           
            # action and states lists for current trajectory
            actions = []
            

            if self.feature_extractor is None:
                state_features = self.env.reset()
            else:
                state_features = self.feature_extractor.extract_features(self.env.reset())
                state_features = torch.from_numpy(state_features).type(torch.FloatTensor).to(DEVICE)
            
            print('Replacing pedestrian :', self.env.cur_ped)
            frames_per_ped = []

            if self.env.replace_subject:
                subject_list.append(self.env.cur_ped)
            if store_raw:
                states = [state]
            else:
                states = [state_features]

            done = False
            t= 0
            unknown_state_counter = 0
            total_states = 0
            run_reward = 0
            
            while not done and t < self.max_episode_length:
                
                action = self.policy.eval_action(state_features)

                if expert_svf is not None:
                    if self.feature_extractor.hash_function(state_features) not in expert_svf.keys():
                        #print(' Unknown state.')
                        unknown_state_counter+=1
                        #pdb.set_trace()
                total_states+=1
                state, rewards, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                    self.feature_extractor.overlay_bins(state)
                run_reward+=rewards

                if self.feature_extractor is not None:
                    state_features = self.feature_extractor.extract_features(state)
                    state_features = torch.from_numpy(state_features).type(torch.FloatTensor).to(DEVICE)

                else:
                    state_features = state

                '''
                print(state[0:9].reshape((3,3)))
                print(state[9:18].reshape((3,3)))
                print(state[18:22])

                print(state[22:134].reshape((16,7)))

                print(state[134:137])
                print(state[137:])
                #pdb.set_trace()
                '''
                if store_raw:
                    states.append(state)
                else:
                    states.append(state_features)

                #take a screenshot if current frame matches the 
                #frame interval

                if capture_frame_interval:

                    if (t%capture_frame_interval==0):

                        screenshot = np.transpose(self.env.take_screenshot(), axes=(1, 0, 2))
                        
                        frames_per_ped.append(screenshot)



                t+=1
            reward_across_trajs.append(run_reward)
            frac_unknown_states_enc.append(unknown_state_counter/total_states)
            if path is not None:

                actions_tensor = torch.tensor(actions)
                if self.feature_extractor and not store_raw:

                    states_tensor = torch.stack(states)
                    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                    torch.save(actions_tensor,
                            os.path.join(path, 'traj%s.acts' % str(traj_i)))
                    torch.save(states_tensor,
                            os.path.join(path, 'traj%s.states' % str(traj_i)))
                
                else:
                    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                    print('Storing for ', traj_i)
                    np.save(os.path.join(path, 'traj%s.states' % str(traj_i)), states)

                #store a screenshot of the end tracing the trajectories of the 
                #ghost and that of the pedestrian
                end_traj_screen_capture = np.transpose(self.env.take_screenshot(), axes=(1, 0, 2))
                frames_per_ped.append(end_traj_screen_capture)
                master_captured_frame_list.append(frames_per_ped)



        return reward_across_trajs, frac_unknown_states_enc, subject_list, master_captured_frame_list




    def reset_training(self):
        """
        Resets the optimizers for the RL 
        """
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)




    def finish_episode(self):
        """Takes care of calculating gradients, updating weights, and resetting
        required variables and histories used in the training cycle one an
        episode ends."""
        R = 0
        saved_actions = self.policy.saved_actions
        saved_entropy_list = self.policy.saved_entropy

        policy_losses = []
        value_losses = []
        entropy_losses = []
        rewards = []

        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards, dtype=torch.float)
        # if single rewards, do not normalize mean distribution
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.EPS)

        for (log_prob, value), r, entropy_val in zip(saved_actions, rewards, saved_entropy_list):
            advantage = r - value.item()
            policy_losses.append(-log_prob * advantage)
            entropy_losses.append(entropy_val)
            r_tensor = torch.tensor([r]).type(torch.float)

            if torch.cuda.is_available():
                r_tensor = r_tensor.cuda()

            #print('value :',value.type(), 'r_tensor :', r_tensor.type())
            #print('value :',value, 'r_tensor :', r_tensor)
            value_losses.append(F.smooth_l1_loss(value, r_tensor))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum() - \
                self.entropy_coeff * torch.stack(entropy_losses).sum()

        #additional lines for loss based termination
        if self.termination is not None:
            self.termination.update_loss_diff_list(loss.item())
            self.termination.plot_avg_loss()
        loss.backward()

        #adding loss in the loss list
        if self.plot_loss or self.save_folder:
            self.loss_mean.append(loss.item())
            if len(self.loss_mean)==self.loss_interval:
                self.loss.append(statistics.mean(self.loss_mean))
                self.loss_mean = []

        #torch.nn.utils.clip_grad_norm_(self.policy.parameters(),.5)
        self.optimizer.step()

        del self.policy.rewards[:]
        del self.policy.saved_actions[:]
        del self.policy.saved_entropy[:]


    def train(self, max_episodes=None, max_episode_length=None, reward_network=None):
        """Train actor critic method on given gym environment."""
        #along with the policy, the train now returns the loss and the 
        #rewards obtained in the form of a list
        running_reward = 0
        running_reward_list =[]
        action_array = np.zeros(self.env.action_space.n)
        plt.figure('Loss')
        if max_episode_length is None:
            max_episode_length = self.max_episode_length
        

        if max_episodes is None:
            max_episodes = self.max_episodes

            
        for i_episode in count(1):

            if self.feature_extractor is not None:

                state = self.feature_extractor.extract_features(
                            self.env.reset())
            else:
                state = self.env.reset()
            state = torch.from_numpy(state).type(torch.FloatTensor).to(DEVICE)
            # number of timesteps taken
            t = 0

            # rewards obtained in this episode
            # ep_reward = self.max_episode_length
            ep_reward = 0

            for t in range(max_episode_length):  # Don't infinite loop while learning

                action = self.policy.sample_action(state)
                action_array[action] += 1
                state, reward, done, _ = self.env.step(action)

                if self.feature_extractor is not None:
                
                    state = self.feature_extractor.extract_features(
                            state)
                    state = torch.from_numpy(state).type(torch.FloatTensor).to(DEVICE)

                if reward_network is None:

                    #print(reward)
                    reward = reward
                else:
                    reward = reward_network(state)
                    reward = reward.item()

                ep_reward += reward

                if self.render:
                    self.env.render()

                self.policy.rewards.append(reward)

                #now does not break when done
                if done:
                    break
                    #pass

            #running_reward = running_reward * self.reward_threshold_ratio +\
            #    ep_reward * (1-self.reward_threshold_ratio)

            running_reward += ep_reward

            self.finish_episode()

            # if not in an IRL setting, solve environment according to specs

            if i_episode >= 10 and i_episode % self.log_interval == 0:
                
                if self.termination is None:
                    print('Ep {}\tLast length: {:5d}\tAvg. reward: {:.2f}'.format(
                        i_episode, t, running_reward/self.log_interval))
                    print('The action frequency array :', action_array)
                    running_reward_list.append(running_reward/self.log_interval)
                    running_reward = 0
                    action_array = np.zeros(self.env.action_space.n)
                else:
                    print('Ep {}\tLast length: {:5d}\tAvg. reward: {:.2f} \
                        \tLoss diff :{:.4f}'.format(
                        i_episode, t, running_reward, 
                        self.termination.current_avg_loss))

                if self.plot_loss:

                    plt.plot(self.loss)
                    plt.draw()
                    plt.pause(.0001)
            

            '''
            commenting the break by 'solving' for now
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                        "the last episode runs to {} time \
                        steps!".format(running_reward, t))
                break
            '''
            # terminate if max episodes exceeded
            if i_episode > max_episodes and max_episodes > 0:

                break


            if self.termination is not None and self.termination.check_termination():
                break

        loss_list = self.loss
        self.loss = []
        self.loss_mean = []

        if self.save_folder:
            self.plot_and_save_info((loss_list, running_reward_list), ('Loss', 'rewards_obtained'))


        return self.policy

    def train_episode(self, reward_acc, reward_network=None, featureExtractor=None):
        """
        performs a single RL training iterations.
        """
        state = self.env.reset()

        # rewards obtained in this episode
        ep_reward = 0

        for t in range(self.max_episode_length):  # Don't infinite loop while learning
            action = self.policy.sample_action(state)
            state, reward, done, _ = self.env.step(action)

            if reward_network is None:
                reward = reward

            else:
                reward = reward_network(state)
                reward = reward.item()

            ep_reward += reward

            g = self.gamma
            reward_acc.value = g * reward_acc.value + (1-g) * ep_reward

            self.policy.rewards.append(reward)

            if done:
                break

        self.finish_episode()

    def train_mp(
        self,
        n_jobs=1,
        reward_net=None,
        feature_extractor=None,
        irl=False,
        log_interval=100
    ):

        self.policy.share_memory()

        # share the reward network memory if it exists
        if reward_net:
            reward_net.share_memory()

        # TODO: The target method here is weirdly setup, where part of the
        # functionality MUST lie outside of any class. How to fix this?
        mp.spawn(
            train_spawnable,
            args=(self, reward_net, irl),
            nprocs=n_jobs
        )

        return self.policy

    def plot_and_save_info(self, inp_tuple, name_tuple):
        #pass a tuple containing n number of lists , this function goes through all and plots them
        i = 0
        color_list  = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r']
        for list_val in inp_tuple:
            plt.figure(name_tuple[i])
            plt.plot(list_val,color_list[i])
            plt.draw()
            plt.pause(.0001)

            #getting the file_name, counting the number of files that are already existing
            folder = self.save_folder + '/' + name_tuple[i] +'/'
            #print('The folder :', folder)
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
            plot_i = 0
            while os.path.exists(os.path.join(folder, '%s.jpg' % plot_i)):
                plot_i += 1

            file_name = folder + str(plot_i)+'.jpg'
            plt.savefig(file_name)
            plt.close()
            i += 1





def train_spawnable(process_index, rl, *args):
    print("%d process spawned." % process_index)
    rl.train(*args)


if __name__ == '__main__':
    args = parser.parse_args()

    _env = gym.make('FrozenLakeNP-v0')
    _env.seed(args.seed)
    torch.manual_seed(args.seed)

    model = ActorCritic(_env, gamma=args.gamma, render=args.render,
                        log_interval=args.log_interval)

    model.train()


    

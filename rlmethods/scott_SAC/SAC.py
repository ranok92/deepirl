import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#from torch.distributions import Normal
import math
import pdb
from rlmethods.scott_SAC import sac_utils
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Returns an action for a given state
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.action_dim = action_dim

    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-4, 15)
        std = log_std.exp()

        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=mean.size())).to(device) 
        action = torch.tanh(z)

        log_prob = (-math.log(2 * math.pi)/self.action_dim - torch.log(std.pow(2) + 1e-6)/self.action_dim - std.pow(-2) * (z - mean).pow(2)) * 0.5
        
        #log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return self.max_action * action, log_prob, self.max_action * torch.tanh(mean)


# Returns a Q-value for given state/action pair
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        q1 = F.relu(self.l1(xu))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(xu))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def q1(self, x, u):
        q1 = F.relu(self.l1(torch.cat([x, u], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x 


class SAC(object):

    def __init__(self, env,
                 feat_extractor=None,
                 policy=None,
                 log_interval=3,
                 max_episodes=1000,
                 max_ep_length=200,
                 hidden_dims=[256],
                 save_folder=None,
                 reward_scale=5):
        
        #adding more parameters to suit the Social navigation project

        self.env = env
        self.feature_extractor = feat_extractor 
        self.log_interval = log_interval


        self.max_ep_length = max_ep_length
        self.max_episodes = max_episodes

        if self.feature_extractor is not None:
            state_dim = self.feature_extractor.extract_features(env.reset()).shape[0]

        else:
            state_dim = env.reset().shape[0] 

        self.hidden_dims = hidden_dims

        self.start_timesteps = 1000
        #**have to fix the action dim thing
        #not using hidden_dims for now

        action_dim = 2
        max_action = 1
        self.orient_quantization = len(self.env.orientation_array)
        self.speed_quantization = len(self.env.speed_array)

        #############################################################

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.value = Value(state_dim).to(device)
        self.value_target = Value(state_dim).to(device)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)

        self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.reward_scale = reward_scale




    def select_action(self, state, test=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.cpu().reshape(1, -1)).to(device)
            action, log_prob, mean_action = self.actor(state)
            
            #if test, select the most likely action
            if test:
                action = mean_action

            action = action.squeeze()
            orient = self.get_quantization(action[0].item(), -1, 2, self.orient_quantization)
            speed = self.get_quantization(action[1].item(), -1, 2, self.speed_quantization)

            action_env = speed*self.orient_quantization+orient

            return action_env


    def get_quantization(self, raw_inp, min_val, max_val, quantization):

        range_val = max_val-min_val
        return int((raw_inp-min_val)/range_val*quantization)


    #the action for the environment is 1 dimensional but it comprises
    #of two things, speed*orientation control.
    #This, in the network is represnted as a 2 dim array
    #but when passing the result into the environment is comverted
    #to a single number by multiplication
    def convert_env_to_network_action(self, env_action):
        #the 2d array represents [orient, speed]
        return np.asarray([env_action%self.orient_quantization, int(env_action/self.orient_quantization)])


    def train(self, reward_network=None, irl=False):

        replay_buffer = sac_utils.ReplayBuffer()
        running_reward = 0
        running_reward_list = []
        total_timesteps = 0
        action_array = np.zeros(self.env.action_space.n)

        for i_episode in count(1):

            if self.feature_extractor is not None:

                state = self.feature_extractor.extract_features(self.env.reset())
            else:
                state = self.env.reset()


            ep_timestep = 0
            ep_reward = 0
            for i in range(self.max_ep_length):

                if total_timesteps < self.start_timesteps:
                    #random action
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state)


                new_state, reward, done, _ = self.env.step(action)
                action_array[action] += 1
                if self.feature_extractor is not None:

                    new_state = self.feature_extractor.extract_features(new_state)

                if reward_network is not None:

                    reward = reward_network(new_state).item()

                ep_reward += reward
                if done:
                    done_bool=1
                else:
                    done_bool=0

                action = self.convert_env_to_network_action(action)
                replay_buffer.add((state.cpu(), new_state.cpu(), action, reward, done_bool))

                if total_timesteps >= self.start_timesteps:

                    self.finish_episode(replay_buffer, 1)

                state = new_state
                ep_timestep += 1
                total_timesteps += 1

                if done:
                    break

            running_reward += ep_reward

            #logging intermediate training information

            if not irl:

                if i_episode >=1 and i_episode %self.log_interval == 0:

                    print('Ep {}\tLast length: {:5d}\tAvg. reward: {:.2f}'.format(
                        i_episode, ep_timestep, running_reward/self.log_interval))
                    print('The action frequency array :', action_array)
                    running_reward_list.append(running_reward/self.log_interval)
                    running_reward = 0
                    action_array = np.zeros(self.env.action_space.n)

                if i_episode > self.max_episodes and self.max_episodes > 0:

                    break

            else:

                assert self.max_episodes > 0

                if i_episode >= 10 and  i_episode % self.log_interval == 0:
                    print('Ep {}\tLast length: {:5d}\tAvg. reward: {:.2f}'.format(
                        i_episode, ep_timestep, running_reward/self.log_interval))
                    print('The action frequency array :', action_array)
                    action_array = np.zeros(self.env.action_space.n)
                    running_reward_list.append(running_reward/self.log_interval)

                    running_reward = 0

                # terminate if max episodes exceeded
                if i_episode > self.max_episodes:
                    break

        if self.save_folder:
            self.plot_and_save_info((loss_list, running_reward_list), ('Loss', 'rewards_obtained'))


        return self.actor



    def finish_episode(self, replay_buffer, iterations, batch_size=256, discount=0.99, tau=0.005): 

        for it in range(iterations):

            # Each of these are batches 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Target Q
            with torch.no_grad():
                target_Q = self.value_target(next_state)
                target_Q = self.reward_scale * reward + done * discount * target_Q

            # Current Q estimate
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q) + self.criterion(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            new_action, log_prob, _ = self.actor(state)
            actor_loss = (log_prob - self.critic.q1(state, new_action)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target V
            with torch.no_grad():
                new_action, log_prob, _ = self.actor(state)
                target_Q1, target_Q2 = self.critic(state, new_action)
                target_V = torch.min(target_Q1, target_Q2) - log_prob

            # Current V estimate
            current_V = self.value(state)

            # Compute value loss
            value_loss = self.criterion(current_V, target_V)

            # Optimize the critic
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)




    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.value.state_dict(), f"{directory}/{filename}_value.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.value.load_state_dict(torch.load(f"{directory}/{filename}_value.pth"))
        self.value_target.load_state_dict(self.value.state_dict())

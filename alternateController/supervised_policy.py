import torch


import torch.nn as nn
import numpy as np 
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import pdb
from tqdm import tqdm
sys.path.insert(0, '..')
from neural_nets.base_network import BasePolicy 
from envs.drone_data_utils import read_training_data

from sklearn.model_selection import train_test_split 
from envs.drone_env_utils import angle_between

from tensorboardX import SummaryWriter


def get_quantization_division(raw_value, quantization_value, num_of_divisions):

    base_division = int(num_of_divisions/2)
    raw_value_to_div = int(raw_value/quantization_value) + base_division
    clipped_value = min(max(0, raw_value_to_div), num_of_divisions)

    return clipped_value


class SupervisedNetwork(BasePolicy):

    def __init__(self, input_size, output_size, hidden_dims=[256]):

        super(SupervisedNetwork, self).__init__()
        self.hidden = []
        self.input = nn.Sequential(
                    nn.Linear(input_size, hidden_dims[0]),
                    nn.ELU()
                    )
        for i in range(1, len(hidden_dims)):

            self.hidden.append(nn.Sequential(
                               nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                               nn.ELU()
                                ))
                    
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(hidden_dims[-1], output_size)
    
    def forward(self, x):

        x = self.input(x)
        for i in range(len(self.hidden)):

            x = self.hidden[i](x)
        
        x = self.output(x)
        return x
    
    def sample_action(self, state):

        x = self.forward(state)
        
        return x
    
    def eval_action(self, state, state_raw, env):

        orient_div = env.orient_quantization
        num_orient_divs = len(env.orientation_array)
        speed_div = env.speed_quantization
        num_speed_divs = len(env.speed_array)
        ref_vector = np.asarray([-1, 0])

        goal_to_agent_vector = state_raw['goal_state'] - state_raw['agent_state']['position']
        current_angle_between = (angle_between(goal_to_agent_vector, ref_vector) - \
                                 angle_between(state_raw['agent_state']['orientation'], ref_vector)
                                )*180/np.pi
        output = self.forward(state)

        change_in_angle = output[0] - current_angle_between
        
        orient_action = get_quantization_division(change_in_angle, orient_div, num_orient_divs)
        change_in_speed = output[1] - state_raw['agent_state']['speed']
        speed_action = get_quantization_division(change_in_speed, speed_div, num_speed_divs)

        return (speed_action*num_orient_divs)+orient_action




class SupervisedPolicy:

    def __init__(self, input_dims, output_dims,
                 hidden_dims=[256],
                 learning_rate=0.001,
                 categorical=True,
                 mini_batch_size=200,
                 save_folder=None):
        '''
        Initialize the class
        '''
        #parameters for the policy network
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_layer = output_dims
        self.policy = SupervisedNetwork(input_dims, output_dims, hidden_dims=self.hidden_dims)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 'cpu')

        self.policy = self.policy.to(self.device)
        self.categorical = categorical
        #parameters for the optimizer
        self.lr = learning_rate
        self.optimizer = optim.SGD(self.policy.parameters(), lr=self.lr)
        if self.categorical:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = torch.nn.MSELoss()
        #parameters for the training
        self.mini_batch_size = mini_batch_size 

        #saving the data
        self.save_folder = save_folder
        self.tensorboard_writer = SummaryWriter(self.save_folder)

    def arrange_data(self, parent_folder, test_data_percent=0.2):
        '''
        loads the data
        '''
        training_data_tensor = read_training_data(parent_folder)
        if self.categorical:
            y_label_size = 1
        else:
            y_label_size = self.output_layer
        x_data = training_data_tensor[:, 0:-y_label_size]
        y_data = training_data_tensor[:, -y_label_size:]
        '''
        if self.categorical:
            y_data_onehot = torch.zeros((y_data.shape[0], self.output_layer)).to(self.device)
            pdb.set_trace()
            y_data_onehot.scatter_(1, y_data, 1)
            y_data = y_data_onehot.type(torch.double)
        '''
        x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                            y_data,
                                                            test_size=test_data_percent)

        return x_train, x_test, y_train, y_test
    
    def train(self, num_epochs):
        '''
        trains a policy network
        '''
        x_train, x_test, y_train, y_test = self.arrange_data('/home/abhisek/Study/Robotics/deepirl/envs/expert_datasets/university_students/annotation/traj_info/frame_skip_1/students003/DroneFeatureRisk_speedv2_with_raw_actions')
        data_loader = DataLoader(torch.cat((x_train, y_train), 1),
                                batch_size=self.mini_batch_size)
        action_counter = 0
        '''
        for i in y_train:
            if i[0]!=17:
                action_counter += 1
        '''
        counter = 0
        if self.categorical:
            label_size = 1
        else:
            label_size = self.output_layer
        for i in tqdm(range(num_epochs)):
            batch_loss = 0
            for batch, sample in enumerate(data_loader):
                x_mini_batch = sample[:, 0:-label_size]
                if self.categorical:
                    y_mini_batch = sample[:, -label_size:].type(torch.long)
                else:
                    y_mini_batch = sample[:, -label_size:].type(torch.float)

                y_pred= self.policy(x_mini_batch.type(torch.float))
                
                loss = self.loss(y_pred, y_mini_batch.squeeze())
                batch_loss += loss
                counter += 1
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            print(batch_loss)
            self.tensorboard_writer.add_scalar('Log_info/loss', batch_loss, i)

        self.tensorboard_writer.close()
        self.policy.save(self.save_folder)


    def play(self, num_runs):
        '''
        Loads up an environment and checks the performance of the agent.
        '''
        


if __name__=='__main__':

    s_policy = SupervisedPolicy(80, 2, categorical=False, hidden_dims=[1024, 4096, 1024], mini_batch_size=2000, save_folder='./Supervised_learning_test')
    s_policy.train(200)


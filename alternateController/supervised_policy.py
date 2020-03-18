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
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from envs.drone_env_utils import angle_between

from tensorboardX import SummaryWriter
from envs.gridworld_drone import GridWorldDrone
import math

from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speedv2

from collections import Counter
from imblearn.over_sampling import RandomOverSampler


def remove_samples(dataset, label_to_remove, no_of_samples):
    '''
    removes samples with label: label_to_remove so that the number of
    samples in the dataset are equal to the value provided in no_of_samples
    '''
    label_counter = Counter(dataset[:, -1])
    total_data = 0
    for val in label_counter:
        if val != label_to_remove:
            total_data += label_counter[val]
    total_data += no_of_samples
    print('Total data :', total_data)
    new_dataset_shape = np.asarray(dataset.shape)
    new_dataset_shape[0] = total_data
    new_dataset_array = np.zeros(new_dataset_shape)
    label_counter = 0
    old_array_counter = 0
    new_array_counter = 0
    for i in range(dataset.shape[0]):
        if dataset[i, -1] == label_to_remove:
            if label_counter < no_of_samples:
                new_dataset_array[new_array_counter, :] = dataset[i, :]
                new_array_counter += 1
                label_counter += 1
                
        else:
            new_dataset_array[new_array_counter, :] = dataset[i, :]
            new_array_counter += 1
    return new_dataset_array


def get_quantization_division(raw_value, quantization_value, num_of_divisions):

    print('raw value :',raw_value)
    print('quantization_value :', quantization_value)
    print('num_of_divisions :', num_of_divisions)
    base_division = int(num_of_divisions/2)
    if raw_value > 0:
        raw_value_to_div = int(raw_value/quantization_value) + base_division
    else:
        raw_value_to_div = math.ceil(raw_value/quantization_value) + base_division
    clipped_value = min(max(0, raw_value_to_div), num_of_divisions-1)
    print ('clipped value :', clipped_value)
    return clipped_value




class SupervisedNetworkRegression(BasePolicy):

    def __init__(self, input_size, output_size, hidden_dims=[256]):

        super(SupervisedNetworkRegression, self).__init__()
        self.hidden = []
        self.input_layer = nn.Sequential(
                    nn.Linear(input_size, hidden_dims[0]),
                    nn.ELU()
                    )
        for i in range(1, len(hidden_dims)):

            self.hidden.append(nn.Sequential(
                               nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                               nn.ELU()
                                ))
                    
        self.hidden_layer = nn.ModuleList(self.hidden)
        
        self.orientation_layer = nn.Sequential(
                                            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                                            nn.ELU(),
                                            nn.Linear(hidden_dims[-1], 1)
                                            )
                                              
        self.speed_layer = nn.Sequential(
                                        nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                                        nn.ELU(),
                                        nn.Linear(hidden_dims[-1], 1)
                                        )

    
    def forward(self, x):

        x = self.input_layer(x)
        for i in range(len(self.hidden)):

            x = self.hidden_layer[i](x)
        
        x_orient = self.orientation_layer(x)
        x_speed = self.speed_layer(x)

        return x_orient, x_speed

        
    
    def sample_action(self, state):

        x = self.forward(state)
        
        return x
    
    
    def eval_action_continuous(self, state, state_raw, env):

        goal_to_agent_vector = state_raw['goal_state'] - state_raw['agent_state']['position']


        signed_angle_between = (np.arctan2(state_raw['agent_state']['orientation'][0],
                                           state_raw['agent_state']['orientation'][1]) - 
                               np.arctan2(goal_to_agent_vector[0],
                                          goal_to_agent_vector[1]))*180/np.pi 

        if signed_angle_between > 180:
            signed_angle_between = signed_angle_between - 360
        elif signed_angle_between < -180:
            signed_angle_between = 360 + signed_angle_between

        output_orient, output_speed = self.forward(state)
        #pdb.set_trace()
        output_orient = output_orient.detach().cpu().numpy()
        output_speed = output_speed.detach().cpu().numpy()

        change_in_angle = output_orient - signed_angle_between
        
        orient_action = min(max(-env.max_orient_change, change_in_angle), 
                            env.max_orient_change)
        change_in_speed = output_speed - state_raw['agent_state']['speed']
        
        speed_action = min(max(-.8, change_in_speed), .8)

        return np.asarray([speed_action, int(orient_action)])




    def eval_action(self, state, state_raw, env):

        orient_div = env.orient_quantization
        num_orient_divs = len(env.orientation_array)
        speed_div = env.speed_quantization
        num_speed_divs = len(env.speed_array)
        ref_vector = np.asarray([-1, 0])

        goal_to_agent_vector = state_raw['goal_state'] - state_raw['agent_state']['position']

        signed_angle_between = (np.arctan2(state_raw['agent_state']['orientation'][0],
                                           state_raw['agent_state']['orientation'][1]) - 
                               np.arctan2(goal_to_agent_vector[0],
                                          goal_to_agent_vector[1]))*180/np.pi 

        if signed_angle_between > 180:
            signed_angle_between = signed_angle_between - 360
        elif signed_angle_between < -180:
            signed_angle_between = 360 + signed_angle_between

        output = self.forward(state)
        output = output.detach().cpu().numpy()
        change_in_angle = output[0] - signed_angle_between
        

        orient_action = get_quantization_division(change_in_angle, orient_div, num_orient_divs)
        change_in_speed =  output[1] - state_raw['agent_state']['speed']
        
        print('The change needed in orientation :{}, change in speed :{}'.format(change_in_angle,
                                                                    change_in_speed))
        speed_action = get_quantization_division(change_in_speed, speed_div, num_speed_divs)
        print('CUrrent heading direction :{}, current s\
peed{}'.format(env.state['agent_head_dir'], env.state['agent_state']['speed']))

        print('The output :', output)
        print('The speed action {}, the orient action {}'.format(speed_action,
                                                                 orient_action))
        #pdb.set_trace()
        return (speed_action*num_orient_divs)+orient_action

class SupervisedNetwork(BasePolicy):

    def __init__(self, input_size, output_size, hidden_dims=[256]):

        super(SupervisedNetwork, self).__init__()
        self.hidden = []
        self.input_layer = nn.Sequential(
                    nn.Linear(input_size, hidden_dims[0]),
                    nn.ELU()
                    )
        for i in range(1, len(hidden_dims)):

            self.hidden.append(nn.Sequential(
                               nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                               nn.ELU()
                                ))
                    
        self.hidden_layer = nn.ModuleList(self.hidden)

        self.output_layer = nn.Linear(hidden_dims[-1], output_size)
    
    def forward(self, x):

        x = self.input_layer(x)
        for i in range(len(self.hidden)):

            x = self.hidden_layer[i](x)
        
        x = self.output_layer(x)
        return x


    
    def sample_action(self, state):

        x = self.forward(state)
        
        return x
    
    
    def eval_action_continuous(self, state, state_raw, env):

        goal_to_agent_vector = state_raw['goal_state'] - state_raw['agent_state']['position']


        signed_angle_between = (np.arctan2(state_raw['agent_state']['orientation'][0],
                                           state_raw['agent_state']['orientation'][1]) - 
                               np.arctan2(goal_to_agent_vector[0],
                                          goal_to_agent_vector[1]))*180/np.pi 

        if signed_angle_between > 180:
            signed_angle_between = signed_angle_between - 360
        elif signed_angle_between < -180:
            signed_angle_between = 360 + signed_angle_between

        output = self.forward(state)
        #pdb.set_trace()
        output = output.detach().cpu().numpy()

        change_in_angle = output[0] - signed_angle_between
        
        orient_action = min(max(-env.max_orient_change, change_in_angle), 
                            env.max_orient_change)
        change_in_speed =  output[1] - state_raw['agent_state']['speed']
        
        speed_action = min(max(-.8, change_in_speed), .8)
        return np.asarray([speed_action, int(orient_action)])




    def eval_action(self, state, state_raw, env):

        orient_div = env.orient_quantization
        num_orient_divs = len(env.orientation_array)
        speed_div = env.speed_quantization
        num_speed_divs = len(env.speed_array)
        ref_vector = np.asarray([-1, 0])

        goal_to_agent_vector = state_raw['goal_state'] - state_raw['agent_state']['position']

        signed_angle_between = (np.arctan2(state_raw['agent_state']['orientation'][0],
                                           state_raw['agent_state']['orientation'][1]) - 
                               np.arctan2(goal_to_agent_vector[0],
                                          goal_to_agent_vector[1]))*180/np.pi 

        if signed_angle_between > 180:
            signed_angle_between = signed_angle_between - 360
        elif signed_angle_between < -180:
            signed_angle_between = 360 + signed_angle_between

        output = self.forward(state)
        output = output.detach().cpu().numpy()
        change_in_angle = output[0] - signed_angle_between
        

        orient_action = get_quantization_division(change_in_angle, orient_div, num_orient_divs)
        change_in_speed =  output[1] - state_raw['agent_state']['speed']
        
        print('The change needed in orientation :{}, change in speed :{}'.format(change_in_angle,
                                                                    change_in_speed))
        speed_action = get_quantization_division(change_in_speed, speed_div, num_speed_divs)
        print('CUrrent heading direction :{}, current s\
peed{}'.format(env.state['agent_head_dir'], env.state['agent_state']['speed']))

        print('The output :', output)
        print('The speed action {}, the orient action {}'.format(speed_action,
                                                                 orient_action))
        #pdb.set_trace()
        return (speed_action*num_orient_divs)+orient_action




class SupervisedPolicy:

    def __init__(self, input_dims, output_dims,
                 hidden_dims=[256],
                 learning_rate=0.001,
                 categorical=True,
                 mini_batch_size=200,
                 policy_path=None,
                 save_folder=None):
        '''
        Initialize the class
        '''
        #parameters for the policy network
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_layer = output_dims

        if not categorical:
            self.policy = SupervisedNetworkRegression(input_dims, output_dims, hidden_dims=self.hidden_dims)
        else:
            self.policy = SupervisedNetwork(input_dims, output_dims, hidden_dims=self.hidden_dims)

        if policy_path is not None:
            self.policy.load(policy_path)
        
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
        self.save_folder = None
        if save_folder:

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

        training_data_numpy = training_data_tensor.cpu().numpy()
        print('Statistics of labels in the original dataset :', Counter(training_data_numpy[:, -1]))

        truncated_training_data = remove_samples(training_data_numpy, 17.0, 1000)
        
        x_data = truncated_training_data[:, 0:-y_label_size]
        y_data = truncated_training_data[:, -y_label_size:]

        print('Statistics of labels after removing extra :', Counter(y_data.squeeze()))


        #remove imbalances from the data in case of categorical data
        if self.categorical:

            ros = RandomOverSampler(random_state=100)
            x_data, y_data = ros.fit_resample(x_data, y_data)
            pdb.set_trace()

            print('The class distribution after upsampling :', Counter(y_data.squeeze()))
                        
        x_data = torch.from_numpy(x_data).to(self.device)
        y_data = torch.from_numpy(np.expand_dims(y_data, axis=1)).to(self.device)


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
    
    def train(self, num_epochs, data_folder):
        '''
        trains a policy network
        '''
        x_train, x_test, y_train, y_test = self.arrange_data(data_folder)
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

                y_pred = self.policy(x_mini_batch.type(torch.float))
                
                loss = self.loss(y_pred, y_mini_batch.squeeze())
                batch_loss += loss
                counter += 1
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            print(batch_loss)
            if self.save_folder:
                self.tensorboard_writer.add_scalar('Log_info/loss', batch_loss, i)
        
        if self.save_folder:
            self.tensorboard_writer.close()
            self.policy.save(self.save_folder)


    def train_regression(self, num_epochs, data_folder):
        '''
        trains a policy network
        '''
        x_train, x_test, y_train, y_test = self.arrange_data(data_folder)
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
            batch_loss_speed = 0
            batch_loss_orient = 0
            for batch, sample in enumerate(data_loader):
                x_mini_batch = sample[:, 0:-label_size]
                if self.categorical:
                    y_mini_batch = sample[:, -label_size:].type(torch.long)
                else:
                    y_mini_batch = sample[:, -label_size:].type(torch.float)

                orient_pred, speed_pred = self.policy(x_mini_batch.type(torch.float))
                #
                loss_orient = self.loss(orient_pred, y_mini_batch.squeeze()[:, 0])
                loss_speed = self.loss(speed_pred, y_mini_batch.squeeze()[:, 1])
                #pdb.set_trace()
                loss = loss_orient + loss_speed

                batch_loss_speed += loss_speed
                batch_loss_orient += loss_orient
                batch_loss += loss

                counter += 1
                print(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            print('Loss from speed :{} , loss from orientation :{} ,batch_loss :{}'.format(batch_loss_speed,
                                                                                           batch_loss_orient,
                                                                                           batch_loss))
            if self.save_folder:
                self.tensorboard_writer.add_scalar('Log_info/loss', batch_loss, i)
                self.tensorboard_writer.add_scalar('Log_info/speed_loss', batch_loss_speed, i)
                self.tensorboard_writer.add_scalar('Log_info/orient_loss', batch_loss_orient, i)
        

        self.tensorboard_writer.close()
        
        if self.save_folder:
            self.policy.save(self.save_folder)



    def play_policy(self,
                    num_runs,
                    max_episode_length,
                    feat_extractor):
        '''
        Loads up an environment and checks the performance of the agent.
        '''
        #initialize variables needed for the run 

        agent_width = 10
        obs_width = 10
        step_size = 2
        grid_size = 10
        
        #load up the environment
        annotation_file = "../envs/expert_datasets/university_students\
/annotation/processed/frame_skip_1/students003_processed_corrected.txt"
        env = GridWorldDrone(
                            display=True,
                            is_onehot=False,
                            seed=0,
                            obstacles=None,
                            show_trail=False,
                            is_random=False,
                            annotation_file=annotation_file,
                            subject=None,
                            tick_speed=60,
                            obs_width=10,
                            step_size=step_size,
                            agent_width=agent_width,
                            replace_subject=True,
                            segment_size=None,
                            external_control=True,
                            step_reward=0.001,
                            show_comparison=True,
                            consider_heading=True,
                            show_orientation=True,
                            continuous_action=True,
                            # rows=200, cols=200, width=grid_size)
                            rows=576,
                            cols=720,
                            width=grid_size,
                        )
        #initialize the feature extractor

        feat_ext = None
        if feat_extractor == "DroneFeatureRisk_speedv2":

            feat_ext = DroneFeatureRisk_speedv2(
                agent_width=agent_width,
                obs_width=obs_width,
                step_size=step_size,
                grid_size=grid_size,
                show_agent_persp=False,
                return_tensor=False,
                thresh1=18,
                thresh2=30,
            )

        #play the environment 

        for i in range(num_runs):
 
            state = env.reset()
            state_features = feat_ext.extract_features(state)
            state_features = torch.from_numpy(state_features).type(torch.FloatTensor).to(self.device)
            done = False
            t = 0
            while t < max_episode_length:

                action = self.policy.eval_action_continuous(state_features, state, env)
                state, _, done, _ = env.step(action)
                state_features = feat_ext.extract_features(state)
                state_features = torch.from_numpy(state_features).type(torch.FloatTensor).to(self.device)
                t+=1
                if done:
                    break



if __name__=='__main__':

    s_policy = SupervisedPolicy(80, 2, 
                                categorical=True, 
                                hidden_dims=[1024, 4096, 1024], 
                                mini_batch_size=2000,
                                #policy_path='./Supervised_learning_test/from_quadra_1.pt',
                                save_folder='./test_new_model')


    data_folder = '../envs/expert_datasets/university_students/annotation/traj_info/frame_skip_1/students003/DroneFeatureRisk_speedv2_with_actions_lag8'
    s_policy.train(10, data_folder)
    #s_policy.train_regression(20)
    #s_policy.play_policy(100, 200, 'DroneFeatureRisk_speedv2')




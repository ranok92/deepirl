import sys
import pdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np 
import math
from tensorboardX import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

sys.path.insert(0, '..')
from neural_nets.base_network import BasePolicy 
from envs.drone_data_utils import read_training_data
from envs.drone_env_utils import angle_between
from envs.gridworld_drone import GridWorldDrone

from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speedv2



def remove_samples(dataset, label_to_remove, no_of_samples):
    '''
    Given a dataset for categorical classification, reduces a particular label as provided to the 
    number of samples entered. 
        input:
            dataset - a dataset in numpy
            label_to_remove - the value of the label to adjust
            no_of_samples - number of samples of that label to retain.

        output:
            truncated_dataset - dataset with number of tuples adjusted as required
            excess_data - the tuples that were removed from the original dataset
    '''
    label_counter = Counter(dataset[:, -1])
    total_tuples_to_retain = 0
    for val in label_counter:
        if val != label_to_remove:
            total_tuples_to_retain += label_counter[val]
    total_tuples_to_retain += no_of_samples
    #print('Total data :', total_tuples_to_retain)
    truncated_dataset_shape = np.asarray(dataset.shape)
    
    truncated_dataset_shape[0] = total_tuples_to_retain
    truncated_dataset_array = np.zeros(truncated_dataset_shape)

    excess_data_shape = truncated_dataset_shape
    excess_data_shape[0] = dataset.shape[0] - total_tuples_to_retain
    excess_data_array = np.zeros(excess_data_shape)

    label_counter = 0
    excess_data_counter = 0
    truncated_array_counter = 0

    for i in range(dataset.shape[0]):
        if dataset[i, -1] == label_to_remove:
            if label_counter < no_of_samples:
                truncated_dataset_array[truncated_array_counter, :] = dataset[i, :]
                truncated_array_counter += 1
                label_counter += 1
            else:
                excess_data_array[excess_data_counter, :] = dataset[i, :]
                excess_data_counter += 1
        else:
            truncated_dataset_array[truncated_array_counter, :] = dataset[i, :]
            truncated_array_counter += 1

    return truncated_dataset_array, excess_data_array


def get_quantization_division(raw_value, quantization_value, num_of_divisions):

    '''
    print('raw value :',raw_value)
    print('quantization_value :', quantization_value)
    print('num_of_divisions :', num_of_divisions)
    '''
    base_division = int(num_of_divisions/2)
    if raw_value > 0:
        raw_value_to_div = int(raw_value/quantization_value) + base_division
    else:
        raw_value_to_div = math.ceil(raw_value/quantization_value) + base_division
    clipped_value = min(max(0, raw_value_to_div), num_of_divisions-1)
    #print ('clipped value :', clipped_value)
    return clipped_value


def rescale_value(value, current_limits, new_limits):
    """
    Given a value and the limits, rescales the value to the new limits
        input:
            value : float variable containing the value
            current_limits : a tuple containing the lower and upper limits 
                             of the value
            new_limits : a tuple containing the desired lower and upper 
                             limits.
    """
    old_range = current_limits[1] - current_limits[0]
    new_range = new_limits[1] - new_limits[0]

    return (value-current_limits[0]) / old_range * new_range \
            + new_limits[0]



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
                                            nn.Sigmoid(),
                                            nn.Linear(hidden_dims[-1], 1)
                                            )
                                              
        self.speed_layer = nn.Sequential(
                                        nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                                        nn.Sigmoid(),
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

        orient_limits = (-30.0, +30.0)
        old_limit = (-1, +1)

        goal_to_agent_vector = state_raw['goal_state'] - state_raw['agent_state']['position']

        signed_angle_between = (np.arctan2(state_raw['agent_state']['orientation'][0],
                                           state_raw['agent_state']['orientation'][1]) - 
                               np.arctan2(goal_to_agent_vector[0],
                                          goal_to_agent_vector[1]))*180/np.pi 

        if signed_angle_between > 180:
            signed_angle_between = signed_angle_between - 360
        elif signed_angle_between < -180:
            signed_angle_between = 360 + signed_angle_between

        orient, speed = self.forward(state)

        orient = orient.detach().cpu().numpy()
        orient_rescale = rescale_value(orient, (0.0, 1.0), (-30, 30))

        speed = speed.detach().cpu().numpy()  
        speed_rescale = rescale_value(speed, (0.0, 1.0), (0.0, 2.0))

        change_in_angle = orient_rescale - signed_angle_between
        

        orient_action = get_quantization_division(change_in_angle, orient_div, num_orient_divs)
        change_in_speed = speed_rescale - state_raw['agent_state']['speed']
        speed_action = get_quantization_division(change_in_speed, speed_div, num_speed_divs)
        '''
        print('The change needed in orientation :{}, change in speed :{}'.format(change_in_angle,
                                                                          change_in_speed))

        print('CUrrent heading direction :{}, current s\
        #peed{}'.format(env.state['agent_head_dir'], env.state['agent_state']['speed']))

        print('The output :', output)
        print('The speed action {}, the orient action {}'.format(speed_action,
                                                                 orient_action))
        pdb.set_trace()
        '''
        return (speed_action * num_orient_divs) + orient_action, \
                np.asarray([orient, speed]), \
                np.asarray([orient_rescale, speed_rescale])



class SupervisedNetworkClassification(BasePolicy):

    def __init__(self, input_size, output_size, hidden_dims=[256]):

        super(SupervisedNetworkClassification, self).__init__()
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
        self.output_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], output_size),
                            )
 


    def forward(self, x):

        x = self.input_layer(x)
        for i in range(len(self.hidden)):

            x = self.hidden_layer[i](x)
        
        x = self.output_layer(x)
        return x


    
    def sample_action(self, state):

        x = self.forward(state)
        
        return x


    def eval_action(self, state_vector):

        output = self.forward(state_vector)
        _, index = torch.max(output, 1)
        return index.unsqueeze(1)




class SupervisedPolicyController:
    '''
    Class to train supervised policies. There are two types of supervised policies, classification based 
    and regression based. 

    Training classification based policies:
        1. set categorical = True
        2. output dims = Number of classes
        3. 
    '''
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
            self.policy = SupervisedNetworkClassification(input_dims, output_dims, hidden_dims=self.hidden_dims)

        if policy_path is not None:
            self.policy.load(policy_path)
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 'cpu')

        self.policy = self.policy.to(self.device)
        self.categorical = categorical
        #parameters for the optimizer
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        if self.categorical:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = torch.nn.MSELoss()
        #parameters for the training
        self.mini_batch_size = mini_batch_size 

        #saving the data
        self.test_interval = 1
        self.save_folder = None
        if save_folder:

            self.save_folder = save_folder
            self.tensorboard_writer = SummaryWriter(self.save_folder)


    def remove_imbalances_from_data(self, training_data_tensor, majority_ratio):
        """
        Takes in a dataset with imbalances in the labels and returns a dataset with relative balance
        in the labels
        input: 
            training_data_tensor : a tensor of shape (no.of samples x size of each sample(including output))
            majority_ratio : a float between 0-1 that denotes how much the non major labels need to be upsampled
                             wrt to the label with the majority.
        
        output:
            x_data : x values of the dataset after balancing as per specifications provided.
            y_data : y values of the dataset after balancing as per specifications provided. 
        """
        majority_label = None
        majority_counts = 0
        training_data_numpy = training_data_tensor.cpu().numpy()
        print('Statistics of labels in the original dataset :', Counter(training_data_numpy[:, -1]))
        original_label_counter = Counter(training_data_numpy[:, -1])
        for val in original_label_counter:
            majority_label = val 
            majority_counts = original_label_counter[val]
            break
        samples_to_retain = int(majority_counts*majority_ratio)
        truncated_training_data, truncated_majority_samples = remove_samples(training_data_numpy, majority_label, samples_to_retain)

        x_data = truncated_training_data[:, 0:-1]
        y_data = truncated_training_data[:, -1:]
        print('Statistics of labels after removing extra :', Counter(y_data.squeeze()))
        #remove imbalances from the data in case of categorical data
        ros = RandomOverSampler(random_state=100)
        x_data, y_data = ros.fit_resample(x_data, y_data)
        x_data = np.concatenate((x_data, truncated_majority_samples[:, 0:-1]), axis=0)
        y_data = np.concatenate((y_data, truncated_majority_samples[:, -1]), axis=0)
        print('The class distribution after upsampling :', Counter(y_data.squeeze()))
        #pdb.set_trace()
        
        return x_data, np.expand_dims(y_data, axis=1)


    def scale_regression_output(self, training_dataset, output_limits):
        '''
        Given the training data, this method scales the data of the output columns
        to be standardized i.e. in a range between 0 and 1.
        input:
            training_dataset: a tensor of shape nxm, where n is the number of tuples in the 
                             dataset and m is the shape of a single tuple including the input and 
                             output
            output_limits : a list of length equal to the number columns in the output which contains
                            tuples denoting the range for values in each of the columns
        
        output:
            scaled_training_dataset: a tensor of shape mxn, where the values in the output columns are 
                                     scaled accordingly.
                    
        '''
        no_of_output_columns = len(output_limits)
    
        output_tensor = training_dataset[:, -no_of_output_columns:]
        input_tensor = training_dataset[:, 0:-no_of_output_columns]
        for i in range(1, no_of_output_columns+1):
            mean_val = output_tensor[:, -i].mean()
            std_val = output_tensor[:, -i].std()
            print("For column: {} \nMean :{}, Std deviation:{}".format(i, 
                                                                       mean_val,
                                                                       std_val))
            min_val = output_limits[-i][0]
            max_val = output_limits[-i][1]
            range_val = max_val - min_val
            output_tensor[:, -i]  = (output_tensor[:, -i] - min_val)/range_val

            mean_val = output_tensor[:, -i].mean()
            std_val = output_tensor[:, -i].std()
            print("After normalization:\n For column: {} \nMean :{}, Std deviation:{}".format(i, 
                                                                mean_val,
                                                                std_val))

        training_dataset[:, -no_of_output_columns:] = output_tensor
        scaled_training_dataset = training_dataset

        return input_tensor.cpu().numpy(), output_tensor.cpu().numpy()






    def arrange_data(self, parent_folder, test_data_percent=0.2):
        '''
        loads the data and arranges it in train test format
            for classification network it handles imbalances in the labels 
            data
            for regression network it scales the output values
        '''
        training_data_tensor = read_training_data(parent_folder)
        
        if self.categorical:
            y_label_size = 1
        else:
            y_label_size = self.output_layer

        if self.categorical:
 
            majority_ratio = .2
            x_data, y_data = self.remove_imbalances_from_data(training_data_tensor,
                                               majority_ratio)
            
        
        else:

            scale_info = [(-180, 180), (0, 2)]
            x_data, y_data = self.scale_regression_output(training_data_tensor, 
                                                     scale_info)
        
        x_data = torch.from_numpy(x_data).to(self.device)
        y_data = torch.from_numpy(y_data).to(self.device)


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
        data_loader_train = DataLoader(torch.cat((x_train, y_train), 1),
                                 shuffle=True,
                                batch_size=self.mini_batch_size)

        data_loader_test = DataLoader(torch.cat((x_test, y_test), 1),
                                      shuffle=True,
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
            epoch_loss = []
            y_train_pred = torch.zeros(y_train.shape)
    
            for batch, sample in enumerate(data_loader_train):
                x_mini_batch = sample[:, 0:-label_size]
                if self.categorical:
                    y_mini_batch = sample[:, -label_size:].type(torch.long)
                else:
                    y_mini_batch = sample[:, -label_size:].type(torch.float)
                
                y_pred_mini_batch = self.policy(x_mini_batch.type(torch.float))
                if (i+1)%self.test_interval == 0:
                    #if the iteration is for eavluation, store the prediction values
                    #pdb.set_trace()

                    y_pred_classes = self.policy.eval_action(x_mini_batch.type(torch.float))
                    y_train_pred[batch*self.mini_batch_size:
                                 batch*self.mini_batch_size+sample.shape[0], :] = y_pred_classes.clone().detach()

                loss = self.loss(y_pred_mini_batch, y_mini_batch.squeeze())
                counter += 1
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss.append(loss.detach())

                if self.save_folder:
                    self.tensorboard_writer.add_scalar('Log_info/loss', loss, i)


            if (i+1)%self.test_interval == 0:
                y_test_pred = torch.zeros(y_test.shape)
                #collecting prediction for test tuples
                for batch, sample in enumerate(data_loader_test):
                    x_mini_batch = sample[:, 0:-label_size]

                    if self.categorical:
                        y_mini_batch = sample[:, -label_size:].type(torch.long)
                    else:
                        y_mini_batch = sample[:, -label_size:].type(torch.float)

                    y_pred_mini_batch = self.policy.eval_action(x_mini_batch.type(torch.float))
                    y_test_pred[batch*self.mini_batch_size:
                                batch*self.mini_batch_size+sample.shape[0], :] = y_pred_mini_batch.clone().detach()

                #pdb.set_trace()
                train_accuracy = accuracy_score(y_train, y_train_pred, normalize=True)
                test_accuracy = accuracy_score(y_test, y_test_pred, normalize=True)
                
                print("For epoch: {} \n Train accuracy :{} | Test accuracy :{}\n=========".format(i,
                                                                                        train_accuracy,
                                                                                    test_accuracy))
                if self.save_folder:

                    self.tensorboard_writer.add_scalar('Log_info/training_accuracy', 
                                                        train_accuracy, i)
                    self.tensorboard_writer.add_scalar('Log_info/testing_accuracy',
                                                        test_accuracy, i)


        
        if self.save_folder:
            self.tensorboard_writer.close()
            self.policy.save(self.save_folder)


    def train_regression(self, num_epochs, data_folder):
        '''
        trains a policy network
        '''
        x_train, x_test, y_train, y_test = self.arrange_data(data_folder)


        data_loader = DataLoader(torch.cat((x_train, y_train), 1),
                                shuffle=True,
                                batch_size=self.mini_batch_size)


        if self.categorical:
            y_train = y_train.type(torch.long)
            y_test = y_test.type(torch.long)
            
        else:
            y_train = y_train.type(torch.float)
            y_test = y_test.type(torch.float)

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

                counter += 1
                #print(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.save_folder:
                    self.tensorboard_writer.add_scalar('Log_info/loss', loss, i)
                    self.tensorboard_writer.add_scalar('Log_info/speed_loss', loss_speed, i)
                    self.tensorboard_writer.add_scalar('Log_info/orient_loss', loss_orient, i)

            if (i+1)%self.test_interval == 0:
                orient_train, speed_train = self.policy(x_train.type(torch.float))
                orient_test, speed_test = self.policy(x_test.type(torch.float))
                
                train_loss = self.loss(orient_train.detach(), y_train.squeeze()[:, 0]) + \
                             self.loss(speed_train.detach(), y_train.squeeze()[:, 1])

                test_loss = self.loss(orient_test.detach(), y_test.squeeze()[:, 0]) + \
                              self.loss(speed_test.detach(), y_test.squeeze()[:, 1])

                print("For epoch: {} \n Training loss :{} | Testing loss :{}\n=========".format(i,
                                                                                        train_loss,
                                                                                        test_loss))

                if self.save_folder:
                    
                    self.tensorboard_writer.add_scalar('Log_info/training_loss', 
                                                        train_loss, i)
                    self.tensorboard_writer.add_scalar('Log_info/testing_loss',
                                                        test_loss.type(torch.float), i)

                
            
            #print('Loss from speed :{} , loss from orientation :{} ,batch_loss :{}'.format(batch_loss_speed,
            #                                                                               batch_loss_orient,
            #                                                                               batch_loss))
            

        

        self.tensorboard_writer.close()
        
        if self.save_folder:
            self.policy.save(self.save_folder)



    def play_policy(self,
                    num_runs,
                    env,
                    max_episode_length,
                    feat_ext):
        '''
        Loads up an environment and checks the performance of the agent.
        '''
        #initialize variables needed for the run 

        agent_width = 10
        obs_width = 10
        step_size = 2
        grid_size = 10
        
        #load up the environment

        #initialize the feature extractor

        #container to store the actions for analysis

        action_raw_list = []
        action_scaled_list = []
        #play the environment 

        for i in range(num_runs):
 
            state = env.reset()
            print("Replacing pedestrian :", env.cur_ped)
            state_features = feat_ext.extract_features(state)
            state_features = torch.from_numpy(state_features).type(torch.FloatTensor).to(self.device)
            done = False
            t = 0
            while t < max_episode_length:
                
                if self.categorical:
                    action = self.policy.eval_action(state_features)
                else:
                    action, raw_action, scaled_action = self.policy.eval_action(state_features, state, env)
                
                action_raw_list.append(raw_action)
                action_scaled_list.append(scaled_action)
                #pdb.set_trace()
                state, _, done, _ = env.step(action)
                state_features = feat_ext.extract_features(state)
                state_features = torch.from_numpy(state_features).type(torch.FloatTensor).to(self.device)
                t+=1
                if done:
                    break
        
        pdb.set_trace()


    def play_regression_policy(self,
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
                            continuous_action=False,
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

                action = self.policy.eval_action(state_features)

                state, _, done, _ = env.step(action)
                state_features = feat_ext.extract_features(state)
                state_features = torch.from_numpy(state_features).type(torch.FloatTensor).to(self.device)
                t+=1
                if done:
                    break



if __name__=='__main__':


    s_policy = SupervisedPolicyController(80, 35, 
                                categorical=True, 
                                hidden_dims=[1024, 4096, 1024], 
                                mini_batch_size=2000,
                                #policy_path='./test_balanced_data_categorical/0.pt',
                                save_folder='./delete_this')

    
    data_folder = '../envs/expert_datasets/university_students/annotation/traj_info/frame_skip_1/\
students003/DroneFeatureRisk_speedv2_with_actions_lag8'
    s_policy.train(10, data_folder)

    '''
    data_folder = '../envs/expert_datasets/university_students/annotation/traj_info/frame_skip_1/\
students003/DroneFeatureRisk_speedv2_with_raw_actions'
    s_policy.train_regression(20, data_folder)
     
    s_policy.play_categorical_policy(100, 200, 'DroneFeatureRisk_speedv2')
  

    '''

import pdb
import torch
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

import numpy as np
import argparse
import torch.multiprocessing as mp
import os

import glob
import copy
import math
from logger.logger import Logger
import matplotlib
import matplotlib.pyplot as plt
import datetime, time
#from debugtools import compile_results
from utils import step_wrapper, reset_wrapper
import copy
import pygame
from alternateController.potential_field_controller import PotentialFieldController as PFController
from alternateController.social_forces_controller import SocialForcesController
from rlmethods.b_actor_critic import ActorCritic
from rlmethods.b_actor_critic import Policy
from tqdm import tqdm
from envs.drone_data_utils import classify_pedestrians
from envs.drone_data_utils import get_pedestrians_in_viscinity
from featureExtractor.drone_feature_extractor import dist_2d


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
#general arguments 

parser.add_argument('--max-ep-length', type=int, default=600, help='Max length of a single episode.')

parser.add_argument('--feat-extractor', type=str, default=None, help='The name of the \
                     feature extractor to be used in the experiment.')


#**************************************************************************#
#arguments related to the environment

parser.add_argument('--annotation-file', type=str, default=None, help='The location of the annotation file to \
                    be used to run the environment.')


#**************************************************************************#
#agent related arguments

parser.add_argument('--agent-type', type=str, default='Potential_field', help='The type of agent to be used to \
                    in the environment. It can be either a RL/IRL agent, or an alternative controller agent. \
                    Different agents will then have different arguments.')

#arguments for a network based agent

parser.add_argument('--policy-path', type=str, nargs='?', default=None)
parser.add_argument('--policy-net-hidden-dims', nargs="*", type=int, default=[128])

#arguments for a potential field agent
'''
/home/abhisek/Study/Robotics/deepirl/experiments/results/Beluga/IRL Runs/
Drone_environment_univ_students003_DroneFeatureRisk_updated_risk_v2_general_3kiter2019-09-27 10:24:41-reg-0-seed-8788-lr-0.001/
saved-models/17.pt
'''

#argument for some other agent



#*************************************************************************#
#parameters for informatio collector

parser.add_argument('--save-folder', type=str, default=None, 
                    help='The name of the folder to \
                    store experiment related information.')

#************************************************************************#

def check_parameters(args):

    """
    checks for validity of the arguments passed to the file.
    """

    if args.agent_type == 'Policy_network':
        if args.policy_path is None or args.policy_net_hidden_dims is None:
            print("Please provide correct information to load a policy network.")
            sys.exit()

        if args.feat_extractor is None:
            print("Please provide a feature extractor to continue.")
            sys.exit()




def agent_drift_analysis(agent,
                        agent_type,
                        env,
                        ped_list,
                        feat_extractor=None,
                        pos_reset=20,
                        ):
    '''
    Performs a drift analysis of a particular agent.
        input:
            agent: the policy/controller
            agent_type: the type of the controller (policy network or an alternate
                        controller like a Potential field or something else)
            env: the environment
            feat_extractor: the feature extractor to be used.
            ped_list: list of pedestrians on which to perform the drift analysis.
            pos_reset: Number of frames after which the position of the agent gets reset
                    to that of the trajectory of the actual pedestrian.
        
        output: the amount of drift incurred by the agent for the pedestrians
            provided in the ped_list.

    '''

    drift_value = 0
    segment_counter = 0
    env.cur_ped = None
    print('Starting drift analysis of agent :{}. Reset\
            interval :{}'.format(agent_type, pos_reset))

    #an array containing the drift value for each pedestrian
    drift_info_detailed = np.zeros(len(ped_list))
    for i in tqdm(range(len(ped_list))):

        #reset the world
        state = env.reset_and_replace(ped=ped_list[i])

        env.goal_state = copy.deepcopy(env.return_position(env.cur_ped, env.current_frame + pos_reset)['position'])
        env.state['goal_state'] = copy.deepcopy(env.goal_state)
        state = copy.deepcopy(env.state)

        final_frame = env.final_frame
        if feat_extractor is not None:
            feat_extractor.reset()
            state_feat = feat_extractor.extract_features(state)
            state_feat = torch.from_numpy(state_feat).type(torch.FloatTensor).to(DEVICE)

            #pass
        #reset the information collector
        done = False
        t = 0
        drift_per_ped = 0
        segment_counter_per_ped = 0
        abs_counter = env.current_frame
        while abs_counter < final_frame:
            #stop_points = []
            if feat_extractor is not None:

                if agent_type == 'Policy_network':
                    action = agent.eval_action(state_feat)
                else:
                #action selection for alternate controller namely potential field
                    action = agent.eval_action(state)

                '''
                if args.render:
                    feat_ext.overlay_bins(state)
                '''
            else:
                action = agent.eval_action(state)
            state, _, done, _ = env.step(action)
            drift_value += dist_2d(env.ghost_state['position'], env.agent_state['position'])
            drift_per_ped += dist_2d(env.ghost_state['position'], env.agent_state['position'])
            if feat_extractor is not None:
                state_feat = feat_extractor.extract_features(state)
                state_feat = torch.from_numpy(state_feat).type(torch.FloatTensor).to(DEVICE)


            t += 1
            abs_counter += 1
            if t%pos_reset == 0:

                segment_counter += 1
                segment_counter_per_ped += 1
                env.agent_state = env.return_position(env.cur_ped, env.current_frame)
                env.state['agent_state'] = copy.deepcopy(env.agent_state)
                '''
                pos = env.agent_state['position']
                stop_points.append(pos)
                for pos in stop_points:
                    pygame.draw.circle(pygame.display.get_surface(),  (0,0,0), (int(pos[1]), int(pos[0])), 20)
                pygame.display.update()
                '''
                env.goal_state = env.return_position(env.cur_ped, env.current_frame + pos_reset)['position']
                env.state['goal_state'] = copy.deepcopy(env.goal_state)
                state = copy.deepcopy(env.state)
                env.release_control = False
                t = 0
                done = False
        
        if segment_counter_per_ped == 0:
            segment_counter_per_ped = 1
        drift_info_detailed[i] = drift_per_ped/segment_counter_per_ped

    return drift_info_detailed



def drift_analysis(agent_list, 
                   agent_type_list,
                   env, 
                   ped_list,
                   feat_extractor=None,
                   start_interval=10, 
                   reset_interval=10, 
                   max_interval=100):
    '''
    input : a list of agents and the reset interval
    returns :
        n lists of size total_ep_length/reset_interval which contains
        the avg total drift value for that agent in that reset value
    '''
    #drift_list is a list that contains arrays which contain drift information of
    #individual pedestrians
    drift_lists = []
    cur_reset_interval = start_interval
    reset_interval_limit = max_interval
    for i in range(len(agent_list)):
        drift_list_per_agent = []
        while cur_reset_interval <= reset_interval_limit:
            drift_list_per_agent.append(agent_drift_analysis(agent_list[i],
                        agent_type_list[i], env, ped_list,
                        feat_extractor=feat_extractor, 
                        pos_reset=cur_reset_interval))
            cur_reset_interval += reset_interval
        drift_lists.append(drift_list_per_agent)
        cur_reset_interval = start_interval
    #plot drift_lists

    return drift_lists
    





if __name__ == '__main__':

    #**************************************************
    step_size = 2
    agent_width = 10
    obs_width = 10
    grid_size = 3

    #**************************************************
    ts=time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    args = parser.parse_args()

    #checks if all the parameters are in order
    check_parameters(args)

    #*************************************************
    #initialize environment
    from envs.gridworld_drone import GridWorldDrone

    consider_heading = True
    env = GridWorldDrone(display=args.render, is_onehot=False,
                        seed=args.seed, obstacles=None,
                        show_trail=True,
                        is_random=False,
                        subject=args.subject,
                        annotation_file=args.annotation_file,
                        tick_speed=60,
                        obs_width=10,
                        step_size=step_size,
                        agent_width=agent_width,
                        external_control=True,
                        replace_subject=args.run_exact,
                        show_comparison=True,
                        consider_heading=consider_heading,
                        show_orientation=True,
                        rows=576, cols=720, width=grid_size)


    print('Environment initalized successfully.')

    #*************************************************
    #initialize the feature extractor

    from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speedv2

    if args.feat_extractor == 'DroneFeatureRisk_speedv2':

        feat_ext = DroneFeatureRisk_speedv2(agent_width=agent_width,
                            obs_width=obs_width,
                            step_size=step_size,
                            grid_size=grid_size,
                            thresh1=18, thresh2=30)

    #*************************************************
    #initialize the agent

    if args.agent_type == 'Policy_network':
        #initialize the network
        print(args.policy_net_hidden_dims)
        print(feat_ext.state_rep_size)
        print(env.action_space)

        agent = Policy(feat_ext.state_rep_size, env.action_space.n, hidden_dims=args.policy_net_hidden_dims)

        if args.policy_path:

            agent.load(args.policy_path)

        else:

            print('Provide a policy path')


    if args.agent_type == 'Potential_field':
        #initialize the PF agent
        max_speed = env.max_speed
        orient_quant = env.orient_quantization
        orient_div = len(env.orientation_array)
        speed_quant = env.speed_quantization
        speed_div = len(env.speed_array)

        attr_mag = 3
        rep_mag = 2
        agent = PFController(speed_div, orient_div, orient_quant)


    if args.agent_type == 'Social_forces':

        orient_quant = env.orient_quantization
        orient_div = len(env.orientation_array)
        speed_quant = env.speed_quantization
        speed_div = len(env.speed_array)
        agent = SocialForcesController(speed_div, orient_div, orient_quant)


    agent_list = []

    #easy, med, hard = classify_pedestrians(args.annotation_file, 30)

    #agent_type_list = ['Potential_field']
    agent_type_list = []
    #agent initialized from the commandline
    agent_file_list = ['/home/abhisek/Study/Robotics/deepirl/experiments/results/Beluga/IRL Runs/Variable-speed-hit-full-run-suppressed-local-updated-features2019-12-14_16:38:00-policy_net-256--reward_net-256--reg-0.001-seed-9-lr-0.0005/saved-models/28.pt']
    agent_file_list.append('/home/abhisek/Study/Robotics/deepirl/experiments/results/Quadra/RL Runs/Possible_strawman2019-12-16 12:22:05DroneFeatureRisk_speedv2-seed-789-policy_net-256--reward_net-128--total-ep-8000-max-ep-len-500/policy-models/0.pt')
    
    for agent_file in agent_file_list:
        
        agent_temp = Policy(feat_ext.state_rep_size, env.action_space.n, hidden_dims=args.policy_net_hidden_dims)
        agent_temp.load(agent_file)
        agent_list.append(agent_temp)
        agent_type_list.append('Policy_network')
    
    start_interval = 50
    reset_int = 30
    reset_lim = 170

    

    #dirft list is list where [[agent1_drift info][agent2_drift_info]]
    #where agent1_dirft_info = [[array containing drift info of peds for a given reset pos]]
    data = np.genfromtxt('./Pedestrian_info/all150.csv', delimiter=' ')
    ped_list = data[:, 1]
    ped_list = ped_list.astype(int)

    ped_list = np.sort(ped_list)
    #ped_list = np.concatenate((easy, med, hard), axis=0)
    ped_list_name = 'all'
    drift_lists = drift_analysis(agent_list, agent_type_list, env,
                                ped_list,
                                feat_extractor=feat_ext,
                                start_interval=start_interval, 
                                reset_interval=reset_int, max_interval=reset_lim)
    
    drift_info_numpy = np.asarray(drift_lists)
    np.save('master_drift_array-50-170-30', drift_info_numpy)
    pdb.set_trace()
    ###################

    x_axis = np.arange(int((reset_lim-start_interval)/reset_int)+1)
    #get the mean and std deviation of pedestrians from drift_lists

    fig, ax = plt.subplots()
    for i in range(len(drift_lists)):
        mean_drift = [np.mean(drift_info_interval) for drift_info_interval in drift_lists[i]]
        std_div_drift = [np.std(drift_info_interval) for drift_info_interval in drift_lists[i]]
        
        ax.errorbar(x_axis, mean_drift, yerr=std_div_drift, label=agent_type_list[i]+str(i),
                    capsize=5, capthick=3, alpha=0.5)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(start_interval+x_axis*reset_int)
    ax.set_xlabel('Reset interval (in frames)')
    ax.set_ylabel('Divergence from ground truth')
    ax.legend()
    plt.show()
    #*******************************************
    '''
    data = np.genfromtxt('./Pedestrian_info/all150.csv', delimiter=' ')
    pdb.set_trace()
    ped_list = data[:,1]
    ped_list = ped_list.astype(int)

    play_environment(ped_list.tolist())
    '''
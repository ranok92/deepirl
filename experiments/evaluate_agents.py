import pdb
import argparse
import sys  # NOQA
import os
import datetime, time
import copy

import numpy as np

import torch
sys.path.insert(0, '..')  # NOQA: E402


from logger.logger import Logger
import matplotlib
import matplotlib.pyplot as plt
#from debugtools import compile_results
from utils import step_wrapper, reset_wrapper
import pygame
from alternateController.potential_field_controller import PotentialFieldController as PFController
from alternateController.social_forces_controller import SocialForcesController
from rlmethods.b_actor_critic import ActorCritic
from rlmethods.b_actor_critic import Policy

from envs.drone_data_utils import classify_pedestrians
from envs.drone_data_utils import get_pedestrians_in_viscinity
from metrics.metrics import (compute_distance_displacement_ratio,
                            compute_trajectory_smoothness,
                            proxemic_intrusions,
                            anisotropic_intrusions)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
#general arguments 

parser.add_argument('--render', action='store_true', help="show the env.")
parser.add_argument('--max-ep-length', type=int, default=600, help='Max length of a single episode.')

parser.add_argument('--feat-extractor', type=str, default=None, help='The name of the \
                     feature extractor to be used in the experiment.')

parser.add_argument('--run-exact', action='store_true')
parser.add_argument('--on-server', action='store_true')

#**************************************************************************#
#arguments related to the environment

parser.add_argument('--annotation-file', type=str, default=None, help='The location of the annotation file to \
                    be used to run the environment.')

parser.add_argument('--reward-path', type=str, nargs='?',
                     default=None)
parser.add_argument('--reward-net-hidden-dims', nargs="*", 
                    type=int, default=[128])

#**************************************************************************#


parser.add_argument(
    "--policy-net-hidden-dims",
    nargs="*",
    type=int,
    default=[128],
    help="The dimensions of the \
                     hidden layers of the policy network.",
)

'''
/home/abhisek/Study/Robotics/deepirl/experiments/results/Beluga/IRL Runs/
Drone_environment_univ_students003_DroneFeatureRisk_updated_risk_v2_general_3kiter2019-09-27 10:24:41-reg-0-seed-8788-lr-0.001/
saved-models/17.pt
'''

#*************************************************************************#
#parameters for informatio collector

parser.add_argument('--save-plots', action='store_true', default=False)
parser.add_argument('--store-results', action='store_true', default=False)

parser.add_argument('--save-folder', type=str, default=None,
                    help='The name of the folder to \
                    store experiment related information.')

def check_parameters(args):
    '''
    Some basic checks on parameters
    '''

    if args.feat_extractor is None:
        print("Please provide a feature extractor to continue.")
        sys.exit()





def main():
    '''
    The main function 
    '''
    #**************************************************
    #parameters for the feature extractors
    thresh1 = 10
    thresh2 = 15

    step_size = 2
    agent_width = 10
    obs_width = 10
    grid_size = 3


    #**************************************************
    #for bookkeeping purposes

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    args = parser.parse_args()

    #checks if all the parameters are in order
    check_parameters(args)

    if args.on_server:

        matplotlib.use('Agg')
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    #*************************************************
    #initialize environment
    from envs.gridworld_drone import GridWorldDrone

    consider_heading = True
    np.random.seed(0)
    env = GridWorldDrone(display=args.render, is_onehot=False,
                        seed=0, obstacles=None,
                        show_trail=True,
                        is_random=False,
                        subject=None,
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
    from featureExtractor.drone_feature_extractor import DroneFeatureRisk, DroneFeatureRisk_v2
    from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speed, DroneFeatureRisk_speedv2
   

    if args.feat_extractor == 'DroneFeatureRisk':

        feat_ext = DroneFeatureRisk(agent_width=agent_width,
                                    obs_width=obs_width,
                                    step_size=step_size,
                                    grid_size=grid_size,
                                    show_agent_persp=True,
                                    thresh1=thresh1, thresh2=thresh2)


    if args.feat_extractor == 'DroneFeatureRisk_v2':

        feat_ext = DroneFeatureRisk_v2(agent_width=agent_width,
                                    obs_width=obs_width,
                                    step_size=step_size,
                                    grid_size=grid_size,
                                    show_agent_persp=False,
                                    thresh1=thresh1, thresh2=thresh2)

    if args.feat_extractor == 'DroneFeatureRisk_speed':

        feat_ext = DroneFeatureRisk_speed(agent_width=agent_width,
                                    obs_width=obs_width,
                                    step_size=step_size,
                                    grid_size=grid_size,
                                    show_agent_persp=True,
                                    thresh1=thresh1, thresh2=thresh2)




    if args.feat_extractor == 'DroneFeatureRisk_speedv2':

        feat_ext = DroneFeatureRisk_speedv2(agent_width=agent_width,
                            obs_width=obs_width,
                            step_size=step_size,
                            grid_size=grid_size,
                            thresh1=18, thresh2=30)

    #*************************************************
    #initialize the agents
    agent_list = [] #list containing the paths to the agents
    agent_type_list = [] #list containing the type of the agents 
    
    #for potential field agent
    attr_mag = 3
    rep_mag = 2

    #agent = PFController()
    ######################
    #for social forces agent

    ######################

    #for network based agents
    agent_file_list = ['/home/abhisek/Study/Robotics/deepirl/experiments/results/Beluga/IRL Runs/Variable-speed-hit-full-run-suppressed-local-updated-features2019-12-14_16:38:00-policy_net-256--reward_net-256--reg-0.001-seed-9-lr-0.0005/saved-models/28.pt']
    agent_file_list.append('/home/abhisek/Study/Robotics/deepirl/experiments/results/Quadra/RL Runs/Possible_strawman2019-12-16 12:22:05DroneFeatureRisk_speedv2-seed-789-policy_net-256--reward_net-128--total-ep-8000-max-ep-len-500/policy-models/0.pt')
    

    #initialize agents based on the agent files
    for agent_file in agent_file_list:
        
        agent_temp = Policy(feat_ext.state_rep_size, 
                            env.action_space.n, 
                            hidden_dims=args.policy_net_hidden_dims)

        agent_temp.load(agent_file)
        agent_list.append(agent_temp)
        agent_type_list.append('Policy_network')

    #####################
    
    for i in range(len(agent_list)):

        while env.cur_ped != env.last_pedestrian:

            state = env.reset()
            done = False
            t = 0
            traj = [copy.deepcopy(state)]
            while not done or t < args.max_ep_length:

                if agent_type_list[i] != 'Policy_Network':

                    feat = feat_ext.extract_features(state)
                    feat = torch.from_numpy(feat).type(torch.FloatTensor).to(DEVICE)

                action = agent_list[i].eval_action(feat)
                state, _ , done, _ = env.step(action)
                traj.append(copy.deepcopy(state))
            
                if done:
                    break

            total_smoothness, avg_smoothness = compute_trajectory_smoothness(traj)
            ratio = compute_distance_displacement_ratio(traj)

            proxemic_intrusions(traj, 10)
            anisotropic_intrusions(traj, 30)
            pdb.set_trace()

if __name__ == "__main__":
    main()
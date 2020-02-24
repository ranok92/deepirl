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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
#general arguments 

parser.add_argument('--render', action='store_true', help="show the env.")
parser.add_argument('--num-trajs', type=int, default=50)
parser.add_argument('--max-ep-length', type=int, default=600, help='Max length of a single episode.')

parser.add_argument('--feat-extractor', type=str, default=None, help='The name of the \
                     feature extractor to be used in the experiment.')

parser.add_argument('--run-exact', action='store_true')
parser.add_argument('--subject', type=int, default=None)
parser.add_argument('--seed', type=int, default=789)
parser.add_argument('--on-server', action='store_true')

#**************************************************************************#
#arguments related to the environment

parser.add_argument('--annotation-file', type=str, default=None, help='The location of the annotation file to \
                    be used to run the environment.')

parser.add_argument('--reward-path' , type=str, nargs='?', default= None)
parser.add_argument('--reward-net-hidden-dims', nargs="*", type=int, default=[128])

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

parser.add_argument('--save-plots', action='store_true', default=False)
parser.add_argument('--store-results', action='store_true', default=False)

parser.add_argument('--save-folder', type=str, default=None, help= 'The name of the folder to \
                    store experiment related information.')

#************************************************************************#

parser.add_argument('--reward-analysis', action='store_true', default=False)
parser.add_argument('--crash-analysis', action='store_true', default=False)
parser.add_argument('--plain-run', action='store_true', default=True)

def check_parameters(args):

    if args.agent_type=='Policy_network':
        if args.policy_path is None or args.policy_net_hidden_dims is None:
            print("Please provide correct information to load a policy network.")
            exit()

        if args.feat_extractor is None:
            print("Please provide a feature extractor to continue.")
            exit()


    if args.reward_analysis:

        if args.reward_path is None or args.reward_net_hidden_dims is None:
            print("Please provide reward network details to perform reward analysis.")
            exit()




#**************************************************
thresh1 = 10
thresh2 = 15

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

if args.on_server:

    matplotlib.use('Agg')
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

#*************************************************
#initialize information collector
from envs.drone_env_utils import InformationCollector

info_collector = InformationCollector(run_info=args.agent_type,
                                      thresh=thresh2*step_size,
                                      plot_info=args.save_plots,
                                      store_info=args.store_results,
                                     )

#*************************************************
#initialize environment
from envs.gridworld_drone import GridWorldDrone

consider_heading = True
np.random.seed(args.seed)
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
from featureExtractor.drone_feature_extractor import DroneFeatureSAM1, DroneFeatureMinimal
from featureExtractor.drone_feature_extractor import DroneFeatureOccup, DroneFeatureRisk
from featureExtractor.drone_feature_extractor import DroneFeatureRisk_v2, DroneFeatureRisk_speed, DroneFeatureRisk_speedv2
if args.feat_extractor == 'DroneFeatureSAM1':

    feat_ext = DroneFeatureSAM1(agent_width=agent_width,
                                obs_width=obs_width,
                                step_size=step_size,
                                grid_size=grid_size,
                                thresh1=thresh1, thresh2=thresh2)

if args.feat_extractor == 'DroneFeatureOccup':

    feat_ext = DroneFeatureOccup(agent_width=agent_width,
                                 obs_width=obs_width,
                                 step_size=step_size,
                                 grid_size=grid_size,
                                 window_size=window_size)


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
#initialize the agent

if args.agent_type == 'Policy_network':
    #initialize the network
    print (args.policy_net_hidden_dims)
    print (feat_ext.state_rep_size)
    print (env.action_space)
    pdb.set_trace()

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


if args.agent_type == 'Default':

    env.external_control = False
    agent = None
    #the person from the video
    pass

#*************************************************
#load reward network if present

if args.reward_path is not None:
    from irlmethods.deep_maxent import RewardNet 

    state_size = feat_ext.extract_features(env.reset()).shape[0]
    reward_net = RewardNet(state_size, args.reward_net_hidden_dims)
    reward_net.load(args.reward_path)

#*************************************************
#play

def reward_analysis():
    '''
    A function to analysis the rewards against actions for a given policy.
    A helpful visualization/ debugging tool
    '''
    for i in range(args.num_trajs):

        #reset the world
        state=env.reset()

        if args.feat_extractor is not None:
            feat_ext.reset()
            state_feat = feat_ext.extract_features(state)
            #pass
        #reset the information collector
        info_collector.reset_info(state)
        done=False
        t = 0
        while t < args.max_ep_length and not done:
            reward_arr = np.zeros(9)
            reward_arr_true = np.zeros(9)

            if args.feat_extractor is not None:

                #************reward analysis block*************
                if args.reward_analysis:
                    
                    for i in range(9): #as there are 9 actions

                        action = i 
                        state, reward_true, _ , _ = env.step(action)
                        print('Taking a step', action)
                        if args.feat_extractor is not None:

                            state_feat_temp = feat_ext.extract_features(state)
                            reward_arr[i] = reward_net(state_feat_temp)
                            reward_arr_true[i] = reward_true
                            state = env.rollback(1)
                            state_feat = feat_ext.rollback(2, state)
                            #print(reward_arr)
                #**********************************************
                #making sure the graphics are consistent

                #if t>0: #skip if this is the first frame
                #    state_feat = feat_ext.extract_features(state)
                #**********************************************
                #selecting the action
                #action selection for network
                if args.agent_type=='Policy_network':
                    #pdb.set_trace()
                    action = agent.eval_action(state_feat)
                else:
                #action selection for alternate controller namely potential field
                    action = agent.eval_action(state)
                #pdb.set_trace()
                #print('The action finally taken :', action)
                #action = int(np.argmax(reward_arr_true))
                #**********************************************

                if args.reward_analysis:
                    #comparing the reward network

                    true_reward_norm = (reward_arr_true - reward_arr_true.mean())/(reward_arr_true.std()+np.finfo(float).eps)
                    network_reward_norm = (reward_arr - reward_arr.mean())/(reward_arr.std()+np.finfo(float).eps)
                    #print('The true reward normalized:\n', true_reward_norm)
                    #print('The network reward normalized: \n', network_reward_norm)
                    plt.plot(true_reward_norm, c='r')
                    plt.plot(network_reward_norm, c='b')
                    plt.plot(probs.cpu().detach().numpy(), c='g')
                    #action = np.argmax(true_reward_norm)
                    #print('Action taken from here:', action)

                    #comparing the policy network


                if args.render:
                    feat_ext.overlay_bins(state)

            else:
                action = agent.eval_action(state) 
            #pdb.set_trace()
            state, reward, done, _ = env.step(action)
           
            if args.feat_extractor is not None:
                state_feat = feat_ext.extract_features(state)
            if args.reward_path is not None:
                reward = reward_net(state_feat)

            #if args.reward_analysis:
            print('Reward : {} for action {}:'.format(reward, action))
                #pdb.set_trace()
            plt.show()
            info_collector.collect_information_per_frame(state)

            t+=1

        info_collector.collab_end_traj_results()

    info_collector.collab_end_results()
    info_collector.plot_information()


def crash_analysis():
    '''
    A visualizing/ debugging tool to analyse with ease the states and conditions
    right before an agent crashes
    '''
    for i in range(args.num_trajs):

        #reset the world
        crash_analysis = False
        state = env.reset()
        print('Current subject :', env.cur_ped)
        if args.feat_extractor is not None:
            feat_ext.reset()
            state_feat = feat_ext.extract_features(state)
            #pass
        #reset the information collector
        done = False
        t = 0
        while t < args.max_ep_length and not done:

            if args.feat_extractor is not None:

            
                if args.agent_type == 'Policy_network':
                    action = agent.eval_action(state_feat)
                else:
                #action selection for alternate controller namely potential field
                    action = agent.eval_action(state)

                if args.render:
                    feat_ext.overlay_bins(state)

            else:
                action = agent.eval_action(state) 
            #pdb.set_trace()
            state, reward_true, done, _ = env.step(action)
            
           
            if args.feat_extractor is not None:
                state_feat = feat_ext.extract_features(state)

                if crash_analysis:
                    pdb.set_trace()
            if args.reward_path is not None:
                reward = reward_net(state_feat)
            else:
                reward = reward_true

            #if args.reward_analysis:
            print('Reward : {} for action {}:'.format(reward, action))
                #pdb.set_trace()

            if done:
                print('Crash frame : ', env.current_frame)
                print('Agent position history :')
                for i in range(len(feat_ext.agent_state_history)):
                    print(feat_ext.agent_state_history[i]['position'], env.heading_dir_history[i])
                if args.crash_analysis:
                    if  reward_true < -0.5:
                        if not crash_analysis:
                            if t > 10:
                                state = env.rollback(10)
                                state_feat = feat_ext.rollback(11, state)
                            else:
                                state = env.rollback(t-1)
                                state_feat = feat_ext.rollback(t, state)
                            print('Current frame after rollback :', env.current_frame)
                            for i in range(len(feat_ext.agent_state_history)):
                                print(feat_ext.agent_state_history[i]['position'], env.heading_dir_history[i])                            
                            done = False
                            crash_analysis = True
                        else:
                            break
                else:
                    break

            t += 1



def agent_drift_analysis(agent=agent,
                        agent_type=args.agent_type,
                        ped_list=None,
                        pos_reset=20,
                        ):
    '''
    if order='by_ped' the drift information is collected per pedestrian
    if order='by_density' the drift information is collected per density of nearby peds 
    step interval after which to reset the position
    input : agent, agent_type and pos_reset.
        Plays the agent on the provided environment with the assigned reset value
        for the assigned number of trajectories. Can be played with or without render
    returns :
        The an array that contains the drift analysis for each of the
        pedestrians in the list for the given pos_reset.
    '''

    drift_value = 0
    segment_counter = 0
    env.cur_ped = None
    print('Starting drift analysis of agent :{}. Reset\
 interval :{}'.format(agent_type, pos_reset))

    if ped_list is not None:
        num_trajs = len(ped_list)
    else:
        num_trajs = args.num_trajs

    #an array containing the drift value for each pedestrian
    drift_info_detailed = np.zeros(num_trajs)
    for i in tqdm(range(num_trajs)):

        #reset the world
        crash_analysis = False
        if ped_list is None:
            state = env.reset()
        else:
            state = env.reset_and_replace(ped=ped_list[i])

        env.goal_state = copy.deepcopy(env.return_position(env.cur_ped, env.current_frame + pos_reset)['position'])
        env.state['goal_state'] = copy.deepcopy(env.goal_state)
        state = copy.deepcopy(env.state)
        #print('Current subject :', env.cur_ped)
        final_frame = env.final_frame
        if args.feat_extractor is not None:
            feat_ext.reset()
            state_feat = feat_ext.extract_features(state)
            state_feat = torch.from_numpy(state_feat).type(torch.FloatTensor).to(DEVICE)

            #pass
        #reset the information collector
        info_collector.reset_info(state)
        done = False
        t = 0
        drift_per_ped = 0
        segment_counter_per_ped = 0
        abs_counter = env.current_frame
        while abs_counter < final_frame:
            stop_points = []
            if args.feat_extractor is not None:

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
            state, reward_true, done, _ = env.step(action)
            drift_value += np.linalg.norm(env.ghost_state['position'] - env.agent_state['position'], 2)
            drift_per_ped += np.linalg.norm(env.ghost_state['position'] - env.agent_state['position'], 2)
            if args.feat_extractor is not None:
                state_feat = feat_ext.extract_features(state)
                state_feat = torch.from_numpy(state_feat).type(torch.FloatTensor).to(DEVICE)

                if crash_analysis:
                    pdb.set_trace()
            if args.reward_path is not None:
                reward = reward_net(state_feat)
            else:
                reward = reward_true


            #info_collector.collect_information_per_frame(state)
            t += 1
            abs_counter += 1
            if t%pos_reset == 0:
                #reset the position of the agent
                #print('t :', t)
                #print('resetting')
                segment_counter += 1
                segment_counter_per_ped += 1
                #print('Drift value : {} for segment {}'.format(drift_value, segment_counter))
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


'''
def agent_drift_analysis_by_density(agent=agent,
                        agent_type=args.agent_type,
                        ped_list=None,
                        viscinity=30,
                        pos_reset=20):
    

    # step interval after which to reset the position
    # input : agent, agent_type, 
    #     ped_list (optional) - list of pedestrians
    #     viscinity (optional) - radius around the agent to get pedestrian density
    #     pos_reset - frames after which to reset the agent

    #     Plays the agent on the provided environment with the assigned reset value
    #     for the assigned number of trajectories. Can be played with or without render
    # returns :
    #     The an array that contains the drift analysis for each of the
    #     pedestrians in the list for the given pos_reset.
    

    drift_value = 0
    segment_counter = 0
    env.cur_ped = None
    print('Starting drift analysis of agent :{}. Reset\
 interval :{}'.format(agent_type, pos_reset))

    if ped_list is not None:
        num_trajs = len(ped_list)
    else:
        num_trajs = args.num_trajs

    #an array containing the drift value for each pedestrian
    drift_info_detailed = {}
    for i in tqdm(range(num_trajs)):

        #reset the world
        crash_analysis = False
        if ped_list is None:
            state = env.reset()
        else:
            state = env.reset_and_replace(ped=ped_list[i])

        env.goal_state = copy.deepcopy(env.return_position(env.cur_ped, env.current_frame + pos_reset)['position'])
        env.state['goal_state'] = copy.deepcopy(env.goal_state)
        state = copy.deepcopy(env.state)
        #print('Current subject :', env.cur_ped)
        final_frame = env.final_frame
        if args.feat_extractor is not None:
            feat_ext.reset()
            state_feat = feat_ext.extract_features(state)
            #pass
        #reset the information collector
        info_collector.reset_info(state)
        done = False
        t = 0
        abs_counter = env.current_frame
        while abs_counter < final_frame:
            stop_points = []
            if args.feat_extractor is not None:

                if agent_type == 'Policy_network':
                    action = agent.eval_action(state_feat)
                else:
                #action selection for alternate controller namely potential field
                    action = agent.eval_action(state)

                
                # if args.render:
                #     feat_ext.overlay_bins(state)
                
            else:
                action = agent.eval_action(state)
            state, reward_true, done, _ = env.step(action)
            #use the state to get the nearby density
            #use the density to classify the frame and store the dirft information
            #accordingly
            
            peds_nearby = get_pedestrians_in_viscinity(state, viscinity)
            drift_value = np.linalg.norm(env.ghost_state['position'] - env.agent_state['position'], 2)
            if str(peds_nearby) in drift_info_detailed.keys():
                drift_info_detailed[str(peds_nearby)] += drift_value
            else:
                drift_info_detailed[str(peds_nearby)] = drift_value
            if args.feat_extractor is not None:
                state_feat = feat_ext.extract_features(state)

                if crash_analysis:
                    pdb.set_trace()
            if args.reward_path is not None:
                reward = reward_net(state_feat)
            else:
                reward = reward_true


            #info_collector.collect_information_per_frame(state)
            t += 1
            abs_counter += 1
            if t%pos_reset == 0:
                #reset the position of the agent
                #print('t :', t)
                #print('resetting')
                segment_counter += 1
                segment_counter_per_ped += 1
                #print('Drift value : {} for segment {}'.format(drift_value, segment_counter))
                env.agent_state = env.return_position(env.cur_ped, env.current_frame)
                env.state['agent_state'] = copy.deepcopy(env.agent_state)
                
                # pos = env.agent_state['position']
                # stop_points.append(pos)
                # for pos in stop_points:
                #     pygame.draw.circle(pygame.display.get_surface(),  (0,0,0), (int(pos[1]), int(pos[0])), 20)
                # pygame.display.update()
                
                env.goal_state = env.return_position(env.cur_ped, env.current_frame + pos_reset)['position']
                env.state['goal_state'] = copy.deepcopy(env.goal_state)
                state = copy.deepcopy(env.state)
                env.release_control = False
                t = 0
                done = False
            
        drift_info_detailed[i] = drift_per_ped/segment_counter_per_ped
    return drift_info_detailed
'''

def play_environment(ped_list):

    agent_type = args.agent_type
    for i in tqdm(range(len(ped_list))):

        #reset the world
        state = env.reset_and_replace(ped=ped_list[i])
        print("Starting pedestrian :", ped_list[i])
        final_frame = env.final_frame
        if args.feat_extractor is not None:
            feat_ext.reset()
            state_feat = feat_ext.extract_features(state)
            state_feat = torch.from_numpy(state_feat).type(torch.FloatTensor).to(DEVICE)

        done = False
        t = 0
        abs_counter = env.current_frame
        while abs_counter < final_frame:
            if args.feat_extractor is not None:

                if agent_type == 'Policy_network':
                    action = agent.eval_action(state_feat)
                elif agent_type == 'Potential_field':
                #action selection for alternate controller namely potential field
                    action = agent.eval_action(state)
                else:
                    action = 0
                '''
                if args.render:
                    feat_ext.overlay_bins(state)
                '''
            else:
                action = agent.eval_action(state)
            state, reward_true, done, _ = env.step(action)
            if args.feat_extractor is not None:
                state_feat = feat_ext.extract_features(state)
                state_feat = torch.from_numpy(state_feat).type(torch.FloatTensor).to(DEVICE)
            
            #info_collector.collect_information_per_frame(state)
            abs_counter += 1


def drift_analysis(agent_list, agent_type_list, 
                   ped_list=None,
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
            drift_list_per_agent.append(agent_drift_analysis(agent_list[i], agent_type_list[i], ped_list=ped_list, pos_reset=cur_reset_interval))
            cur_reset_interval += reset_interval
        drift_lists.append(drift_list_per_agent)
        cur_reset_interval = start_interval
    #plot drift_lists

    return drift_lists
    





if __name__ == '__main__':

    
    '''
    agent_drift_analysis(80)
    '''
    #**************** performing reward analysis
    '''
    reward_analysis()
    '''
    #************ performing drift analysis **************
    
    #initialize the agents
    #for potential field agent
    attr_mag = 3
    rep_mag = 2
    #agent = PFController()

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
    drift_lists = drift_analysis(agent_list, agent_type_list, ped_list=ped_list, start_interval=start_interval, reset_interval=reset_int, max_interval=reset_lim)
    
    drift_info_numpy = np.asarray(drift_lists)
    np.save('master_drift_array-50-170-30', drift_info_numpy)
    pdb.set_trace()
    ###################
    '''
    for i in range(drift_info_numpy.shape[1]):
        drift_info_particular_segment = drift_info_numpy[:, i, :]
        drift_diff = drift_info_particular_segment[0, :] - drift_info_particular_segment[1, :]
        drift_diff_frac = np.divide(drift_diff, drift_info_particular_segment[0, :])
        drift_diff_frac_with_ped = np.concatenate((np.expand_dims(drift_diff_frac, 1),
                                                   np.expand_dims(ped_list, 1)),
                                                   axis=1)
        sort_index = np.argsort(drift_diff_frac)
        sort_diff_and_ped = drift_diff_frac_with_ped[sort_index, :]
        x_axis_2 = np.arange(drift_info_numpy.shape[-1])

        plt.bar(x_axis_2, sort_diff_and_ped[:,0])
        plt.xticks(x_axis_2, sort_diff_and_ped[:, 1])
        file_name = ped_list_name + str(start_interval) +'.csv'
        #np.savetxt(file_name, sort_diff_and_ped)
        plt.show()
        pdb.set_trace()
    '''
    #####################
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
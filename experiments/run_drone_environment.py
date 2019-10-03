import pdb
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

import numpy as np
import argparse
import torch.multiprocessing as mp
import os

import glob
from logger.logger import Logger
import matplotlib
import datetime, time
#from debugtools import compile_results
from utils import step_wrapper, reset_wrapper

parser = argparse.ArgumentParser()
#general arguments 
parser.add_argument('--render', action='store_true', help="show the env.")
parser.add_argument('--num-trajs', type=int, default=50)
parser.add_argument('--max-ep-length', type=int, default=600, help='Max length of a single episode.')

parser.add_argument('--feat-extractor', type=str, default=None, help='The name of the \
                     feature extractor to be used in the experiment.')

parser.add_argument('--run-exact', action='store_true')
parser.add_argument('--seed', type=int, default=789)
parser.add_argument('--on-server', action='store_true')

#**************************************************************************#
#arguments related to the environment

parser.add_argument('--annotation-file', type=str, default='../envs/expert_datasets/university_students/annotation/processed/frame_skip_1/students003_processed_corrected.txt', help='The location of the annotation file to \
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


#argument for some other agent



#*************************************************************************#
#parameters for informatio collector

parser.add_argument('--save-plots', action='store_true', default=False)
parser.add_argument('--store-results', action='store_true', default=False)

parser.add_argument('--save-folder', type=str, default=None, help= 'The name of the folder to \
                    store experiment related information.')

#************************************************************************#

def main():

    #**************************************************
    thresh1 = 15
    thresh2 = 30

    step_size = 2
    agent_width = 10
    obs_width = 10
    grid_size = 10


    #**************************************************
    ts=time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    args = parser.parse_args()

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
    if args.agent_type =='Potential_field':
        consider_heading = False
    env = GridWorldDrone(display=args.render, is_onehot = False, 
                        seed=args.seed, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        annotation_file=args.annotation_file,
                        tick_speed=60, 
                        obs_width=10,
                        step_size=step_size,
                        agent_width=agent_width,
                        train_exact=args.run_exact,
                        show_comparison=True,
                        consider_heading=consider_heading,
                        show_orientation=True,
                        #rows=200, cols=300, width=grid_size)                       
                        rows=576, cols=720, width=grid_size)


    print('Environment initalized successfully.')

    #*************************************************
    #initialize the feature extractor
    from featureExtractor.drone_feature_extractor import DroneFeatureSAM1, DroneFeatureMinimal
    from featureExtractor.drone_feature_extractor import DroneFeatureOccup, DroneFeatureRisk
    from featureExtractor.drone_feature_extractor import DroneFeatureRisk_v2
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


    #*************************************************
    #initialize the agent

    if args.agent_type=='Policy_network':
        #initialize the network
        from rlmethods.b_actor_critic import ActorCritic

        agent = ActorCritic(env, feat_extractor=feat_ext,  gamma=1,
                            log_interval=100,max_ep_length=args.max_ep_length,
                            hidden_dims=args.policy_net_hidden_dims,
                            )

        if args.policy_path:

            agent.policy.load(args.policy_path)

        else:

            print('Provide a policy path')


    if args.agent_type=='Potential_field':
        #initialize the PF agent
        from alternateController.potential_field_controller import PotentialFieldController as PFController

        attr_mag = 5
        rep_mag = 3
        agent = PFController(attr_f_limit=attr_mag,
                             rep_f_limit=rep_mag)

    if args.agent_type=='Default':

        #the person from the video
        pass

    #*************************************************
    #play
    for i in range(args.num_trajs):

        #reset the world
        state=env.reset()
        if args.feat_extractor is not None:
            state_feat = feat_ext.extract_features(state)
            #pass
        #reset the information collector
        info_collector.reset_info(state)
        done=False
        t = 0
        while t < args.max_ep_length and not done:

            if args.feat_extractor is not None:
                action = agent.select_action(state_feat)
                if args.render:
                    feat_ext.overlay_bins(state)

            else:
                action = agent.select_action(state) 
            #pdb.set_trace()
            state, reward, done, _ = env.step(action)
            info_collector.collect_information_per_frame(state)

            if args.feat_extractor is not None:
                state_feat = feat_ext.extract_features(state)
            t+=1

        info_collector.collab_end_traj_results()

    info_collector.collab_end_results()
    info_collector.plot_information()


if __name__ == '__main__':
    main()
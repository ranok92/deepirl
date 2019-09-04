import pdb
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

import numpy as np
import argparse
import torch.multiprocessing as mp
import os

from logger.logger import Logger
import matplotlib
import datetime, time

from utils import step_wrapper, reset_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--policy-path', type=str, nargs='?', default=None)
parser.add_argument('--reward-path' , type=str, nargs='?', default= None)
parser.add_argument('--play', action='store_true',
                    help='play given or latest stored policy.')
parser.add_argument('--play-user' , action='store_true' , 
                    help='lets the user play the game')
parser.add_argument('--dont-save', action='store_true',
                   help="don't save the policy network weights.")
parser.add_argument('--render', action='store_true', help="show the env.")
parser.add_argument('--num-trajs', type=int, default=10)
parser.add_argument('--view-reward', action='store_true')

parser.add_argument('--policy-net-hidden-dims', nargs="*", type=int, default=[128])
parser.add_argument('--reward-net-hidden-dims', nargs="*", type=int, default=[128])

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--on-server', action='store_true')

parser.add_argument('--feat-extractor', type=str, default=None, help='The name of the \
                     feature extractor to be used in the experiment.')
parser.add_argument('--save-folder', type=str, default=None, help= 'The name of the folder to \
                    store experiment related information.')

parser.add_argument('--annotation-file', type=str, default='../envs/expert_datasets/data_zara/annotation/processed/crowds_zara01_processed.txt', 
                    help='The location of the annotation file to be used to run the environment.')

parser.add_argument('--total-episodes', type=int, default=1000, help='Total episodes of RL')
parser.add_argument('--max-ep-length', type=int, default=200, help='Max length of a single episode.')

parser.add_argument('--train-exact', acton='store_true')


def main():
    
    #####for the logger
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    ###################

    args = parser.parse_args()

    if args.on_server:

        matplotlib.use('Agg')
        # pygame without monitor
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    mp.set_start_method('spawn')

    from rlmethods.b_actor_critic import ActorCritic
    from envs.gridworld_drone import GridWorldDrone
    from featureExtractor.drone_feature_extractor import DroneFeatureSAM1
    from featureExtractor.gridworld_featureExtractor import FrontBackSide,LocalGlobal,OneHot,SocialNav,FrontBackSideSimple

    save_folder = None
    if not args.dont_save:

        if not args.save_folder:
            print('Provide save folder.')
            exit()

        save_folder = './results/'+ args.save_folder +st + args.feat_extractor + \
                      '-total-ep-'+str(args.total_episodes)+'-max-ep-len-'+ str(args.max_ep_length)

        experiment_logger = Logger(save_folder,'experiment_info.txt')
        experiment_logger.log_header('Arguments for the experiment :')
        experiment_logger.log_info(vars(args))
    

    agent_width = 10
    step_size = 2
    obs_width = 10
    grid_size = 10

    #initialize the feature extractor to be used
    if args.feat_extractor == 'Onehot':
        feat_ext = OneHot(grid_rows = 10 , grid_cols = 10)
    if args.feat_extractor == 'SocialNav':
        feat_ext = SocialNav(fieldList = ['agent_state','goal_state'])
    if args.feat_extractor == 'FrontBackSideSimple':
        feat_ext = FrontBackSideSimple(thresh1 = 1,
                                    thresh2 = 2,
                                    thresh3 = 3,
                                    thresh4=4,
                                    step_size=step_size,
                                    agent_width=agent_width,
                                    obs_width=obs_width,
                                    )

    if args.feat_extractor == 'LocalGlobal':
        feat_ext = LocalGlobal(window_size=5, grid_size=grid_size,
                           agent_width=agent_width, 
                           obs_width=obs_width,
                           step_size=step_size,
                           )
    
    if args.feat_extractor == 'DroneFeatureSAM1':

        feat_ext = DroneFeatureSAM1(agent_width=agent_width,
                                    obs_width=obs_width,
                                    step_size=step_size,
                                    grid_size=grid_size,
                                    thresh1=5, thresh2=10)
    

    #log feature extractor info

    if not args.dont_save:

        experiment_logger.log_header('Parameters of the feature extractor :')
        experiment_logger.log_info(feat_ext.__dict__)

    #initialize the environment
    if args.train_exact:
        train_exact=True
    else:
        train_exact=False
    env = GridWorldDrone(display=args.render, is_onehot = False, 
                        seed=999, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        annotation_file=args.annotation_file,
                        subject=None,
                        tick_speed=90, 
                        obs_width=10,
                        step_size=step_size,
                        agent_width=agent_width,
                        train_exact=train_exact,
                        show_comparison=True,                       
                        rows=576, cols=720, width=grid_size)

    #log environment info
    if not args.dont_save:

        experiment_logger.log_header('Environment details :')
        experiment_logger.log_info(env.__dict__)

    #initialize RL 
    model = ActorCritic(env, feat_extractor=feat_ext,  gamma=1,
                        log_interval=10,max_ep_length=args.max_ep_length,
                        hidden_dims=args.policy_net_hidden_dims,
                        save_folder=save_folder, 
                        lr=args.lr,
                        max_episodes = args.total_episodes)

    #log RL info
    if not args.dont_save:

        experiment_logger.log_header('Details of the RL method :')
        experiment_logger.log_info(model.__dict__)
    


    if args.policy_path is not None:

        policy_file_list =  []
        reward_across_models = []
        if os.path.isfile(args.policy_path):
            policy_file_list.append(args.policy_path)
        if os.path.isdir(args.policy_path):
            policy_names = glob.glob(os.path.join(args.policy_path, '*.pt'))
            policy_file_list = sorted(policy_names, key=numericalSort)

        xaxis = np.arange(len(policy_file_list))


        
    if not args.play and not args.play_user:
        if args.reward_path is None:
            model.train()
        else:
            from irlmethods.deep_maxent import RewardNet
            state_size = feat_ext.extract_features(env.reset()).shape[0]
            reward_net = RewardNet(state_size, args.reward_net_hidden_dims)
            reward_net.load(args.reward_path)
            print(next(reward_net.parameters()).is_cuda)
            model.train(reward_net = reward_net)

        if not args.dont_save:  
            model.policy.save(save_folder+'/policy-models/')


    if args.play:
        #env.tickSpeed = 15
        xaxis = []
        counter = 1
        print(policy_file_list)
        for policy_file in policy_file_list:

            model.policy.load(policy_file)

            env.tickSpeed = 15
            assert args.policy_path is not None, 'pass a policy to play from!'

            reward_across_models.append(model.generate_trajectory(args.num_trajs, args.render))

        #model.generate_trajectory(args.num_trajs, save_folder+'/agent_generated_trajectories/')

    if args.play_user:
        env.tickSpeed = 200

        model.generate_trajectory_user(args.num_trajs, './user_generated_trajectories/')

if __name__ == '__main__':
    main()

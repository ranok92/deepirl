import pdb
import os

import argparse
import matplotlib
import numpy as np
import sys, time # NOQA
sys.path.insert(0, '..')  # NOQA: E402
from envs.gridworld_drone import GridWorldDrone as GridWorld
from logger.logger import Logger
import utils
from tensorboardX import SummaryWriter

from featureExtractor.drone_feature_extractor import DroneFeatureSAM1, DroneFeatureRisk, DroneFeatureRisk_v2
from featureExtractor.gridworld_featureExtractor import FrontBackSide,LocalGlobal,OneHot,SocialNav,FrontBackSideSimple
from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speed, DroneFeatureRisk_speedv2


import datetime
from logger.logger import Logger

parser = argparse.ArgumentParser()

#***************arguments for environment initialization and general training 

parser.add_argument('--seed', type=int, default=7, help='The seed for the run')

parser.add_argument('--save-folder', type=str, default=None, required=True,
                    help='The name of the directory to store the results in. The name will be used to \
                    save the plots, the policy and the reward networks.(Relative path)')

parser.add_argument('--exp-trajectory-path', type=str, default=None, required=True,
                    help='The name of the directory in which \
                    the expert trajectories are stored.(Relative path)')
parser.add_argument('--annotation-file', type=str, default=None, help='The location of the annotation file to \
                    be used to run the environment.')

parser.add_argument('--replace-subject', action='store_true', default=None)

parser.add_argument('--segment-size', type=int, default=None, help='Size of each trajectory segment.')

parser.add_argument('--subject', type=int, default=None)


parser.add_argument('--dont-save', action='store_true',
                    help="don't save the policy network weights.")

parser.add_argument('--render', action='store_true', help="show the env.")

parser.add_argument('--on-server', action='store_true',                    
                    help="True if the code is being run on a server.")


parser.add_argument('--irl-iterations', type=int, default=100)

#***************************arguments for the feature extractor

parser.add_argument('--feat-extractor', type=str, 
                    default=None, required=True,
                    help='The name of the \
                    feature extractor to be used in the experiment.')

#****************************arguments for the rl 

parser.add_argument("--replay-buffer-size", type=int)
parser.add_argument("--replay-buffer-sample-size", type=int)
parser.add_argument("--log-alpha", type=float, default=-2.995)
parser.add_argument("--entropy-target", type=float, default=0.008)
parser.add_argument("--max-episode-length", type=int, default=10 ** 2)
parser.add_argument("--play-interval", type=int, default=1)
parser.add_argument("--rl-episodes", type=int, default=10 ** 2)

parser.add_argument('--policy-net-hidden-dims', nargs="*", type=int , default=[256], help='The dimensions of the \
                     hidden layers of the policy network.')


#******************************arguments for the IRL 

parser.add_argument('--regularizer', type=float, default=0, help='The regularizer to use.')

parser.add_argument('--reward-net-hidden-dims', nargs="*", type=int , default=[128], 
                     help='The dimensions of the \
                     hidden layers of the reward network.')

parser.add_argument('--lr-irl', type=float, default=1e-3, 
                    help='The learning rate for the reward network.')

parser.add_argument('--clipping-value', type=float, default=None, 
                    help='For gradient clipping of the \
                    reward network.')

parser.add_argument('--scale-svf', action='store_true', default=None, 
                    help='If true, will scale the states \
                    based on the reward the trajectory got.')


def main():
    args = parser.parse_args()

    if args.on_server:
        # matplotlib without monitor
        matplotlib.use('Agg')

        # pygame without monitor
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    #####for the logger
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    ###################

    if not args.save_folder:
        print('Provide save folder.')
        exit()

    policy_net_dims = '-policy_net-'
    for dim in args.policy_net_hidden_dims:
        policy_net_dims += str(dim) 
        policy_net_dims += '-'

    reward_net_dims = '-reward_net-'
    for dim in args.reward_net_hidden_dims:
        reward_net_dims += str(dim)
        reward_net_dims += '-'


    parent_dir = './results/'+str(args.save_folder)+st+policy_net_dims + reward_net_dims
    to_save = './results/'+str(args.save_folder)+st+policy_net_dims + reward_net_dims + \
              '-reg-'+str(args.regularizer)+ \
              '-seed-'+str(args.seed)+'-lr-'+str(args.lr_irl)
                
                
    log_file = 'Experiment_info.txt'

    experiment_logger = Logger(to_save, log_file)
    experiment_logger.log_header('Arguments for the experiment :')
    experiment_logger.log_info(vars(args))


    #from rlmethods.rlutils import LossBasedTermination
    #for rl
    from rlmethods.b_actor_critic import ActorCritic
    from rlmethods.soft_ac_pi import SoftActorCritic
    from rlmethods.rlutils import ReplayBuffer


    #for irl
    from irlmethods.deep_maxent import DeepMaxEnt
    import irlmethods.irlUtils as irlUtils
    from featureExtractor.gridworld_featureExtractor import OneHot,LocalGlobal,SocialNav,FrontBackSideSimple

    agent_width = 10
    step_size = 2
    obs_width = 10
    grid_size = 10


    if args.feat_extractor is None:

        print('Feature extractor missing.')
        exit()
    
    #check for the feature extractor being used
    #initialize feature extractor
    if args.feat_extractor == 'Onehot':
        feat_ext = OneHot(grid_rows = 10 , grid_cols = 10)
    if args.feat_extractor == 'SocialNav':
        feat_ext = SocialNav()
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


    if args.feat_extractor == 'DroneFeatureRisk':
        
        feat_ext = DroneFeatureRisk(agent_width=agent_width,
                                    obs_width=obs_width,
                                    step_size=step_size,
                                    grid_size=grid_size,
                                    thresh1=15, thresh2=30)

    if args.feat_extractor == 'DroneFeatureRisk_v2':
        
        feat_ext = DroneFeatureRisk_v2(agent_width=agent_width,
                                    obs_width=obs_width,
                                    step_size=step_size,
                                    grid_size=grid_size,
                                    thresh1=15, thresh2=30)


    if args.feat_extractor == 'DroneFeatureRisk_speed':

        feat_ext = DroneFeatureRisk_speed(agent_width=agent_width,
                            obs_width=obs_width,
                            step_size=step_size,
                            grid_size=grid_size,
                            thresh1=10, thresh2=15)


    if args.feat_extractor == 'DroneFeatureRisk_speedv2':

        feat_ext = DroneFeatureRisk_speedv2(agent_width=agent_width,
                            obs_width=obs_width,
                            step_size=step_size,
                            grid_size=grid_size,
                            thresh1=18, thresh2=30)


    experiment_logger.log_header('Parameters of the feature extractor :')
    experiment_logger.log_info(feat_ext.__dict__)


    #initialize the environment
    if not args.dont_save and args.save_folder is None:
        print('Specify folder to save the results.')
        exit()


    '''
    environment can now initialize without an annotation file
    if args.annotation_file is None:
        print('Specify annotation file for the environment.')
        exit()
    '''
    if args.exp_trajectory_path is None:
        print('Specify expert trajectory folder.')
        exit()

    #**set is_onehot to false
    goal_state = np.asarray([1,5])
    '''
    env = GridWorld(display=args.render, is_onehot= False,is_random=False,
                    rows =10,
                    cols =10,
                    seed = 7,
                    obstacles = [np.asarray([5,5])],
                                
                    goal_state = np.asarray([1,5]))

    '''

    env = GridWorld(display=args.render, is_random=True,
                    rows=576, cols=720,
                    agent_width=agent_width,
                    step_size=step_size,
                    obs_width=obs_width,
                    width=grid_size,
                    subject=args.subject,
                    annotation_file=args.annotation_file,
                    goal_state=goal_state, 
                    step_wrapper=utils.step_wrapper,
                    seed=args.seed,
                    replace_subject=args.replace_subject,
                    segment_size=args.segment_size,
                    external_control=True,
                    continuous_action=True,
                    reset_wrapper=utils.reset_wrapper,
                    consider_heading=True,
                    is_onehot=False)
    

    experiment_logger.log_header('Environment details :')
    experiment_logger.log_info(env.__dict__)


    #CHANGE HEREq

    #CHANGE HERE
    #initialize loss based termination
    # intialize RL method
    #CHANGE HERE

    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    tbx_writer = SummaryWriter(to_save)
    rl_method = SoftActorCritic(env, replay_buffer,
                                feat_ext,
                                buffer_sample_size=args.replay_buffer_sample_size,
                                tbx_writer=tbx_writer,
                                entropy_tuning=True,
                                tau=0.005,
                                log_alpha=args.log_alpha,
                                entropy_target=args.entropy_target,
                                render=args.render,
                                checkpoint_interval=100000000,
                                play_interval=args.play_interval,
                                )


    print("RL method initialized.")
    print(rl_method.policy)

    experiment_logger.log_header('Details of the RL method :')
    experiment_logger.log_info(rl_method.__dict__)
    

    # initialize IRL method
    #CHANGE HERE 
    trajectory_path = args.exp_trajectory_path

    if args.scale_svf is None:
        scale = False
    
    if args.scale_svf:
        scale = args.scale_svf
    irl_method = DeepMaxEnt(trajectory_path, 
                           rlmethod=rl_method, 
                           rl_episodes=args.rl_episodes,
                           env=env,
                           iterations=args.irl_iterations,
                           on_server=args.on_server,
                           l1regularizer=args.regularizer,
                           learning_rate=args.lr_irl,
                           seed=args.seed,
                           graft=False,
                           scale_svf=scale,
                           rl_max_ep_len=args.max_episode_length,
                           hidden_dims=args.reward_net_hidden_dims,
                           clipping_value=args.clipping_value,
                           enumerate_all=True,
                           save_folder=parent_dir)

    print("IRL method intialized.")
    print(irl_method.reward)

    experiment_logger.log_header('Details of the IRL method :')
    experiment_logger.log_info(irl_method.__dict__)
    irl_method.train()

    if not args.dont_save:
        pass


if __name__ == '__main__':
    main()

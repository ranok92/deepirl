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

from featureExtractor.drone_feature_extractor import DroneFeatureSAM1
from featureExtractor.gridworld_featureExtractor import FrontBackSide,LocalGlobal,OneHot,SocialNav,FrontBackSideSimple

import datetime
from logger.logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--policy-path', type=str, nargs='?', default=None)
parser.add_argument('--play', action='store_true',
                    help='play given or latest stored policy.')
parser.add_argument('--dont-save', action='store_true',
                    help="don't save the policy network weights.")
parser.add_argument('--render', action='store_true', help="show the env.")
parser.add_argument('--on-server', action='store_true',
                    help="True if the code is being run on a server.")
parser.add_argument('--store-train-results', action='store_true',
                    help='True if you want to store intermediate results')
parser.add_argument('--store-interval', action='store_true',
                    help='Interval of storing the results.')
parser.add_argument('--rl-episodes', type=int, default=50)
parser.add_argument('--rl-ep-length', type=int, default=30)
parser.add_argument('--irl-iterations', type=int, default=100)
parser.add_argument('--rl-log-intervals', type=int, default=10)

parser.add_argument('--regularizer', type=float, default=0, help='The regularizer to use.')

parser.add_argument('--seed', type=int, default=7, help='The seed for the run')

parser.add_argument('--save-folder', type=str, default=None,
                    help='The name of the directory to store the results in. The name will be used to \
                    save the plots, the policy and the reward networks.(Relative path)')

parser.add_argument('--exp-trajectory-path', type=str, default=None, help='The name of the directory in which \
                    the expert trajectories are stored.(Relative path)')

parser.add_argument('--feat-extractor', type=str, default=None, help='The name of the \
                     feature extractor to be used in the experiment.')

parser.add_argument('--reward-net-hidden-dims', nargs="*", type=int , default=[128], help='The dimensions of the \
                     hidden layers of the reward network.')

parser.add_argument('--policy-net-hidden-dims', nargs="*", type=int , default=[128], help='The dimensions of the \
                     hidden layers of the policy network.')

parser.add_argument('--annotation-file', type=str, default='../envs/expert_datasets/university_ \
                    students/annotation/processed/frame_skip_1/students003_processed.txt', help='The location of the annotation file to \
                    be used to run the environment.')

parser.add_argument('--lr-rl', type=float, default=1e-3, help='The learning rate for the policy network.')

parser.add_argument('--lr-irl', type=float, default=1e-3, help='The learning rate for the reward network.')

parser.add_argument('--clipping-value', type=float, default=None, help='For gradient clipping of the \
                    reward network.')

parser.add_argument('--scale-svf', action='store_true', default=None, help='If true, will scale the states \
                    based on the reward the trajectory got.')

parser.add_argument('--train-exact', action='store_true', default=None)
parser.add_argument('--subject', type=int, default=None)

#IMPORTANT*** search for 'CHANGE HERE' to find that most probably need changing
#before running on different settings
def main():
    args = parser.parse_args()

    if args.on_server:
        # matplotlib without monitor
        matplotlib.use('Agg')

        # pygame without monitor
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    #####for the logger
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    ###################

    parent_dir = './results/'+str(args.save_folder)+st
    to_save = './results/'+str(args.save_folder)+st+'-reg-'+str(args.regularizer)+ \
                '-seed-'+str(args.seed)+'-lr-'+str(args.lr_rl)
    log_file = 'Experiment_info.txt'

    experiment_logger = Logger(to_save, log_file)
    experiment_logger.log_header('Arguments for the experiment :')
    experiment_logger.log_info(vars(args))


    #from rlmethods.rlutils import LossBasedTermination
    from rlmethods.b_actor_critic import ActorCritic
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

    experiment_logger.log_header('Parameters of the feature extractor :')
    experiment_logger.log_info(feat_ext.__dict__)


    #initialize the environment
    if not args.dont_save and args.save_folder is None:
        print('Specify folder to save the results.')
        exit()

    if args.annotation_file is None:
        print('Specify annotation file for the environment.')
        exit()

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
                    train_exact=args.train_exact,
                    reset_wrapper=utils.reset_wrapper,
                    is_onehot=False)
    


    experiment_logger.log_header('Environment details :')
    experiment_logger.log_info(env.__dict__)


    #CHANGE HEREq

    #CHANGE HERE
    #initialize loss based termination
    # intialize RL method
    #CHANGE HERE
    rlMethod = ActorCritic(env, gamma=1,
                            log_interval=args.rl_log_intervals,
                            max_episodes=args.rl_episodes,
                            max_ep_length=args.rl_ep_length,
                            termination=None,
                            save_folder=to_save,
                            lr=args.lr_rl,
                            hidden_dims=args.policy_net_hidden_dims,
                            feat_extractor=feat_ext)
    print("RL method initialized.")
    print(rlMethod.policy)
    if args.policy_path is not None:
        rlMethod.policy.load(args.policy_path)


    experiment_logger.log_header('Details of the RL method :')
    experiment_logger.log_info(rlMethod.__dict__)
    

    # initialize IRL method
    #CHANGE HERE 
    trajectory_path = args.exp_trajectory_path

    irlMethod = DeepMaxEnt(trajectory_path, 
                           rlmethod=rlMethod, 
                           env=env,
                           iterations=args.irl_iterations,
                           on_server=args.on_server,
                           regularizer=args.regularizer,
                           learning_rate=args.lr_irl,
                           seed=args.seed,
                           graft=False,
                           scale_svf=args.scale_svf,
                           hidden_dims = args.reward_net_hidden_dims,
                           clipping_value=args.clipping_value,
                           save_folder=parent_dir)
    print("IRL method intialized.")
    print(irlMethod.reward)

    experiment_logger.log_header('Details of the IRL method :')
    experiment_logger.log_info(irlMethod.__dict__)
    rewardNetwork = irlMethod.train()

    if not args.dont_save:
        pass


if __name__ == '__main__':
    main()

import pdb
import argparse
import matplotlib
import numpy as np
import gym, sys, time, os

import torch
import numpy as np 

import datetime
sys.path.insert(0, '..')  # NOQA: E402
from logger.logger import Logger
import utils
from mountain_car import extract_features

from mountain_car import MCFeatures, MCFeaturesOnehot

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

parser.add_argument('--reward-net-hidden-dims', nargs="*", type=int , default=[128], help='The dimensions of the \
                     hidden layers of the reward network.')

parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate for the reward network.')


parser.add_argument('--feat-extractor', type=str, default='MCFeatures', help='The feature extractor  \
                    to be used in the experiment')

parser.add_argument('--state-discretization', type=int, default=128, help='The number of discrete \
                    parts you want to break the state')

parser.add_argument('--scale-svf', action='store_true', default=None, help='If true, will scale the states \
                    based on the reward the trajectory got.')
#IMPORTANT*** search for 'CHANGE HERE' to find that most probably need changing
#before running on different settings

'''
python run_experiment.py --on-server --rl-episodes 1000 --rl-ep-length 200 --irl-itreations 200 --rl-log-intervals 100
                        --seed 100 --exp-trajectory-path './exp_traj_mountain_car/' --reward-net-hidden-dims 256
'''


def main():
    args = parser.parse_args()

    if args.on_server:
        # matplotlib without monitor
        matplotlib.use('Agg')

        # pygame without monitor
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    ###
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    ###
    parent_dir = './results/'+str(args.save_folder)+st
    to_save = './results/'+str(args.save_folder)+st+'-reg-'+str(args.regularizer)+ \
              '-seed-'+str(args.seed)+'-lr-'+str(args.lr)
              #'-seed-'+str(args.seed)
    log_file = 'Experiment_info.txt'
    experiment_logger = Logger(to_save, log_file)

    experiment_logger.log_header('Arguments for the experiment :')
    experiment_logger.log_info(vars(args))

    from rlmethods.rlutils import LossBasedTermination
    from rlmethods.b_actor_critic import ActorCritic
    from irlmethods.deep_maxent import DeepMaxEnt
    import irlmethods.irlUtils as irlUtils


    
    #check for the feature extractor being used
    #initialize feature extractor
    if args.feat_extractor=='MCFeatures':
        feat_ext = MCFeatures(args.state_discretization, args.state_discretization) 

    if args.feat_extractor=='MCFeaturesOnehot':
        feat_ext = MCFeaturesOnehot(args.state_discretization, args.state_discretization)

    experiment_logger.log_header('Parameters of the feature extractor :')
    experiment_logger.log_info(feat_ext.__dict__)


    #initialize the environment
    if not args.dont_save and args.save_folder is None:
        print('Specify folder to save the results.')
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
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    #pdb.set_trace()
    experiment_logger.log_header('Environment details :')
    experiment_logger.log_info(env.__dict__)


    #CHANGE HEREq

    #CHANGE HERE
    #initialize loss based termination
    # intialize RL method
    #CHANGE HERE
    rlMethod = ActorCritic(env, gamma=0.99,
                            log_interval=args.rl_log_intervals,
                            max_episodes=args.rl_episodes,
                            max_ep_length=args.rl_ep_length,
                            termination=None,
                            hidden_dims=args.reward_net_hidden_dims,
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

    irlMethod = DeepMaxEnt(trajectory_path, rlmethod=rlMethod, env=env,
                           iterations=args.irl_iterations, log_intervals=5,
                           on_server=args.on_server,
                           regularizer = args.regularizer,
                           learning_rate = args.lr,
                           graft=True,
                           seed=args.seed,
                           scale_svf=args.scale_svf,
                           hidden_dims = args.reward_net_hidden_dims,
                           save_folder=parent_dir)
    print("IRL method intialized.")
    experiment_logger.log_header('Details of the IRL method :')
    experiment_logger.log_info(irlMethod.__dict__)
    
    print(irlMethod.reward)
    rewardNetwork = irlMethod.train()

    if not args.dont_save:
        pass


if __name__ == '__main__':
    main()

import pdb
import os

import argparse
import matplotlib
import numpy as np
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402
from envs.gridworld_clockless import GridWorldClockless as GridWorld
import utils


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


#IMPORTANT*** search for 'CHANGE HERE' to find that most probably need changing
#before running on different settings
def main():
    args = parser.parse_args()

    if args.on_server:
        # matplotlib without monitor
        matplotlib.use('Agg')

        # pygame without monitor
        os.environ['SDL_VIDEODRIVER'] = 'dummy'


    from rlmethods.rlutils import LossBasedTermination
    from rlmethods.b_actor_critic import ActorCritic
    from irlmethods.deep_maxent import DeepMaxEnt
    import irlmethods.irlUtils as irlUtils
    from featureExtractor.gridworld_featureExtractor import OneHot,LocalGlobal,SocialNav,FrontBackSideSimple
    # initialize the environment

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
    env = GridWorld(display=args.render, is_random=False,
                    rows = 10, cols = 10,
                    obstacles = [np.asarray([5,5])],
                    goal_state=goal_state, 
                    step_wrapper=utils.step_wrapper,
                    seed=args.seed,
                    reset_wrapper=utils.reset_wrapper,
                    is_onehot = False)
    
    #CHANGE HERE
    #initialize feature extractor
    #feat_ext = OneHot(grid_rows = 10 , grid_cols = 10)
    #feat_ext = SocialNav(fieldList = ['agent_state','goal_state'])
    feat_ext = LocalGlobal(window_size=3, 
                           fieldList = ['agent_state','goal_state','obstacles'])
    #feat_ext = FrontBackSideSimple(thresh1 = 1,
    #                                thresh2 = 2,
    #                                thresh3 = 3,
    #                                fieldList = ['agent_state','goal_state','obstacles'])
    #CHANGE HERE
    #initialize loss based termination



    # intialize RL method
    #CHANGE HERE
    #pass the appropriate feature extractor
    rlMethod = ActorCritic(env, gamma=0.99,
                            log_interval = args.rl_log_intervals,
                            max_episodes=args.rl_episodes,
                            max_ep_length=args.rl_ep_length,
                            termination = None,
                            feat_extractor = feat_ext)
    print("RL method initialized.")
    if args.policy_path is not None:
        rlMethod.policy.load(args.policy_path)

    

    # initialize IRL method
    #CHANGE HERE 
    trajectory_path = './trajs/ac_gridworld_rectified_loc_glob_window_3/'
    save_plot = './plots/Svf_dict_seed_'+str(args.seed)+'/'

    if os.path.exists(save_plot):
        pass
    else:
        os.mkdir(save_plot)

    irlMethod = DeepMaxEnt(trajectory_path, rlmethod=rlMethod, env=env,
                           iterations=args.irl_iterations, log_intervals=5,
                           on_server=args.on_server,
                           regularizer = args.regularizer,
                           plot_save_folder=save_plot)
    print("IRL method intialized.")
    rewardNetwork = irlMethod.train()

    if not args.dont_save:
        pass


if __name__ == '__main__':
    main()

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
parser.add_argument('--rl-log-intervals', type=int, default=100)


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
    from featureExtractor.gridworld_featureExtractor import OneHot
    # initialize the environment

    #**set is_onehot to false
    env = GridWorld(display=args.render, obstacles=[np.asarray([1, 2])], 
                   step_wrapper=utils.step_wrapper,
                   seed = 3,
                    reset_wrapper=utils.reset_wrapper,
                    is_onehot = False)

    #initialize feature extractor

    feat_ext = OneHot(grid_rows = 10 , grid_cols = 10)

    #initialize loss based termination

    lbt = LossBasedTermination(list_size = 80, stop_threshold = 1.5 , info= False)
    # intialize RL method
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
    trajectory_path = './trajs/ac_gridworld/'
    irlMethod = DeepMaxEnt(trajectory_path, rlmethod=rlMethod, env=env,
                           iterations=args.irl_iterations, log_intervals=5,
                           on_server=args.on_server,
                           plot_save_folder='./plots/')
    print("IRL method intialized.")
    rewardNetwork = irlMethod.train()

    if not args.dont_save:
        pass


if __name__ == '__main__':
    main()

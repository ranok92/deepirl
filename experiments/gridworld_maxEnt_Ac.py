import pdb
import argparse

import numpy as np
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

from envs.gridworld import GridWorld
from rlmethods.b_actor_critic import ActorCritic
from irlmethods.deep_maxent import DeepMaxEnt
import irlmethods.irlUtils as irlUtils

parser = argparse.ArgumentParser()
parser.add_argument('--policy-path', type=str, nargs='?', default=None)
parser.add_argument('--play', action='store_true',
                    help='play given or latest stored policy.')
parser.add_argument('--dont-save', action='store_true',
                    help="don't save the policy network weights.")
parser.add_argument('--render', action='store_true', help="show the env.")


def main():
    args = parser.parse_args()

    # initialize the environment
    env = GridWorld(display=args.render, obstacles=[np.asarray([1, 2])])

    # intialize RL method
    rlMethod = ActorCritic(env, gamma=0.99, log_interval=100,
                           max_episodes=3000, max_ep_length=30)

    if args.policy_path is not None:
        rlMethod.policy.load(args.policy_path)

    # initialize IRL method
    trajectory_path = './trajs/ac_gridworld/'
    irlMethod = DeepMaxEnt(trajectory_path, rlmethod=rlMethod, env=env,
                           iterations=100, log_intervals=5)

    rewardNetwork = irlMethod.train()

    if not args.dont_save:
        rewardNetwork.save('./saved-models-rewards/')

    # if args.play:
        # env.display = True
        # env.tickSpeed = 15
        # assert args.policy_path is not None, 'pass a policy to play from!'
        # rlMethod.generate_trajectory(1000, './trajs/ac_gridworld/')


if __name__ == '__main__':
    main()
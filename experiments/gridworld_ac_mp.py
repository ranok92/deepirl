import pdb
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

import numpy as np
import argparse
import torch.multiprocessing as mp
from envs.gridworld_clockless import GridWorldClockless as GridWorld
from rlmethods.b_actor_critic import ActorCritic

from utils import step_wrapper, reset_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--policy-path', type=str, nargs='?', default=None)
parser.add_argument('--play', action='store_true',
                    help='play given or latest stored policy.')
parser.add_argument('--dont-save', action='store_true',
                   help="don't save the policy network weights.")
parser.add_argument('--render', action='store_true', help="show the env.")
parser.add_argument('--num-trajs', type=int, default=10)

def main():
    args = parser.parse_args()
    mp.set_start_method('spawn')

    env = GridWorld(display=args.render, obstacles=[np.asarray([1, 2])],
                    reset_wrapper=reset_wrapper, step_wrapper=step_wrapper)

    model = ActorCritic(env, gamma=0.99, log_interval=100, max_episodes=10**4,
                        max_ep_length=30)

    if args.policy_path is not None:
        model.policy.load(args.policy_path)

    if not args.play:
        model.train_mp(n_jobs=4)

        if not args.dont_save:
            model.policy.save('./saved-models/')

    if args.play:
        env.tickSpeed = 15
        assert args.policy_path is not None, 'pass a policy to play from!'

        model.generate_trajectory(args.num_trajs, './trajs/ac_gridworld/')

if __name__ == '__main__':
    main()

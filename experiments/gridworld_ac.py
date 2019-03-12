import pdb
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402
import utils

import numpy as np
import argparse

from envs.gridworld_clockless import GridWorldClockless as GridWorld
from rlmethods.b_actor_critic import ActorCritic

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


    env = GridWorld(
        display=args.render,
        obstacles=[np.asarray([1, 2])],
        step_wrapper=utils.step_wrapper,
        reset_wrapper=utils.reset_wrapper,
        stepReward = .01
    )

    model = ActorCritic(env, gamma=0.99, log_interval=100, max_episodes=10**4,
                        max_ep_length=30)

    if args.policy_path is not None:
        model.policy.load(args.policy_path)

    if not args.play:
        model.train()

        if not args.dont_save:
            model.policy.save('./saved-models/')

    if args.play:
        env.tickSpeed = 15
        assert args.policy_path is not None, 'pass a policy to play from!'

        model.generate_trajectory(args.num_trajs, './trajs/ac_gridworld/')

if __name__ == '__main__':
    main()

import pdb
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

from rlmethods.rlutils import LossBasedTermination as LBT
import numpy as np
import argparse
import torch.multiprocessing as mp
from rlmethods.b_actor_critic import ActorCritic
from utils import reset_wrapper, step_wrapper
from irlmethods.deep_maxent import RewardNet

parser = argparse.ArgumentParser()
parser.add_argument('--policy-path', type=str, nargs='?', default=None)
parser.add_argument('--reward-net', type=str, nargs='?', default=None)
parser.add_argument('--play', action='store_true',
                    help='play given or latest stored policy.')
parser.add_argument('--dont-save', action='store_true',
                   help="don't save the policy network weights.")
parser.add_argument('--render', action='store_true', help="show the env.")
parser.add_argument('--num-trajs', type=int, default=10)
parser.add_argument('--irl', action='store_true')

def main():
    args = parser.parse_args()

    if args.render:
        from envs.gridworld import GridWorld
    else:
        from envs.gridworld_clockless import GridWorldClockless as GridWorld

    env = GridWorld(
        display=args.render,
        obstacles=[np.asarray([1, 2])],
        step_wrapper=step_wrapper,
        reset_wrapper=reset_wrapper,
        seed = 3
    )
    loss_t = LBT(list_size=100,stop_threshold=.2,log_interval=50)
    model = ActorCritic(env, gamma=0.99, log_interval=100, max_episodes=5000,
                        max_ep_length=20, termination = loss_t)

    if args.policy_path is not None:
        model.policy.load(args.policy_path)

    if args.reward_net is not None:
        reward_net = RewardNet(env.reset().shape[0])
        reward_net.to('cuda')
        reward_net.load('./saved-models-rewards/0.pt')
        reward_net.eval()
    else:
        reward_net = None

    if not args.play:
        model.train_mp(n_jobs=4, reward_net=reward_net,
                        irl=args.irl)

        if not args.dont_save:
            model.policy.save('./saved-models/')

    if args.play:
        env.tickSpeed = 15
        assert args.policy_path is not None, 'pass a policy to play from!'

        model.generate_trajectory(args.num_trajs, './trajs/ac_gridworld/')

if __name__ == '__main__':
    main()

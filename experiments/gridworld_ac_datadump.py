import pdb
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

import numpy as np
import argparse
import torch.multiprocessing as mp
from envs.gridworld_clockless import GridWorldClockless as GridWorld
from rlmethods.b_actor_critic import ActorCritic

from utils import step_wrapper, reset_wrapper
from rlmethods.termination import DataDumperTermination

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=10**4)
parser.add_argument('--njobs', type=int, default=4)

def main():
    args = parser.parse_args()
    mp.set_start_method('spawn')

    env = GridWorld(display=False, obstacles=[np.asarray([1, 2])],
                    reset_wrapper=reset_wrapper, step_wrapper=step_wrapper)

    data_dumper_termination = DataDumperTermination(args.max_episodes)

    model = ActorCritic(env, gamma=0.99, log_interval=100, max_episodes=10**4,
                        max_ep_length=30, termination=data_dumper_termination)

    model.train_mp(n_jobs=args.njobs)

if __name__ == '__main__':
    main()

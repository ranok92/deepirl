import sys
import numpy as np
import torch
from tensorboardX import SummaryWriter
sys.path.insert(0, '..')  # NOQA: E402

from rlmethods.soft_ac import SoftActorCritic
from rlmethods.soft_ac import DEVICE
from envs.EWAP_gridworld import EwapGridworld
from argparse import ArgumentParser

import gym

parser = ArgumentParser()
parser.add_argument('replay_buffer_size', type=int)
parser.add_argument('replay_buffer_sample_size', type=int)
parser.add_argument('--log-alpha', type=float, default=-2.995)
parser.add_argument('--max-episodes', type=int, default=10**4)
parser.add_argument('--play-interval', type=int, default=100)

args = parser.parse_args()


def main():

    tbx_writer = SummaryWriter(comment='_alpha_' + str(args.log_alpha))

    env = EwapGridworld(ped_id=6)

    soft_ac = SoftActorCritic(
        env,
        replay_buffer_size=args.replay_buffer_size,
        buffer_sample_size=args.replay_buffer_sample_size,
        tbx_writer=tbx_writer,
        tau=0.005,
        log_alpha=args.log_alpha,
        entropy_tuning=False,
    )

    soft_ac.train_and_play(args.max_episodes, args.play_interval)

if __name__ == "__main__":
    main()

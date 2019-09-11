import sys
import numpy as np
import torch
from tensorboardX import SummaryWriter
sys.path.insert(0, '..')  # NOQA: E402

from rlmethods.soft_ac import SoftActorCritic
from rlmethods.soft_ac import DEVICE
from envs.simple_gw import SimpleGridworld
from argparse import ArgumentParser

import gym

parser = ArgumentParser()
parser.add_argument('replay_buffer_size', type=int)
parser.add_argument('replay_buffer_sample_size', type=int)
parser.add_argument('--log-alpha', type=float, default=-2.995)

args = parser.parse_args()


def main():

    tbx_writer = SummaryWriter(comment='alpha_'+str(args.log_alpha))

    env = gym.make('CartPole-v0')

    soft_ac = SoftActorCritic(
        env,
        replay_buffer_size=args.replay_buffer_size,
        buffer_sample_size=args.replay_buffer_sample_size,
        tbx_writer = tbx_writer,
        tau=0.005,
        log_alpha=args.log_alpha,
        entropy_tuning=False,
    )

    soft_ac.train_and_play(10**4, 1)

if __name__ == "__main__":
    main()

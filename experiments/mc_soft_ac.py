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

args = parser.parse_args()

def play(rl, gw):
    done = False

    state = gw.reset()
    total_reward = 0
    iters = 0
    while not done:
        _state = torch.from_numpy(state).type(torch.float).to(DEVICE)
        action, _, _ = rl.select_action(_state)
        next_state, reward, done, _ = gw.step(action.item())

        # update environment variables
        total_reward += reward
        state = next_state

        iters += 1

    return total_reward


def main():

    tbx_writer = SummaryWriter()
    breakpoint()

    env = gym.make('MountainCar-v0')

    soft_ac = SoftActorCritic(
        env,
        replay_buffer_size=args.replay_buffer_size,
        buffer_sample_size=args.replay_buffer_sample_size,
        tbx_writer = tbx_writer,
    )

    for i in range(10**6):
        soft_ac.train()

        if i % 1000 == 0:
            rs = []
            for j in range(10):
                rs.append(play(soft_ac, env))

            tbx_writer.add_scalar('avg reward', np.mean(rs), i)

        if i% 10000 == 0:
            soft_ac.replay_buffer.buffer.clear()


if __name__ == "__main__":
    main()

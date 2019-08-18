import sys
import numpy as np
import torch
from tensorboardX import SummaryWriter
sys.path.insert(0, '..')  # NOQA: E402

from rlmethods.soft_ac import SoftActorCritic
from rlmethods.soft_ac import DEVICE
from envs.simple_gw import SimpleGridworld

def play(rl, gw):
    done = False

    state = torch.from_numpy(gw.reset()).to(DEVICE)
    total_reward = 0
    iters = 0
    while not done and iters < 30:
        action, _ = rl.select_action(state)
        next_state, reward, done, _ = gw.step(action)
        total_reward += reward

        iters += 1

    return total_reward


def main():

    tbx_writer = SummaryWriter()

    env = SimpleGridworld((10, 10), np.array([5, 5]), np.array([7, 7]))
    soft_ac = SoftActorCritic(
        env,
        replay_buffer_size=10**5,
        buffer_sample_size=10**4,
        tbx_writer = tbx_writer,
    )

    for i in range(10**6):
        soft_ac.train()

        if i % 100 == 0:
            rs = []
            for j in range(5):
                rs.append(play(soft_ac, env))

            tbx_writer.add_scalar('avg reward', np.mean(rs), i)

        if i% 10000 == 0:
            soft_ac.replay_buffer.buffer.clear()


if __name__ == "__main__":
    main()

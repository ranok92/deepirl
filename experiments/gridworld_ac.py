import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

import numpy as np

from envs.gridworld import GridWorld
from rlmethods.b_actor_critic import ActorCritic


def main():
    env = GridWorld(display=False, obstacles=[np.asarray([1, 2])])

    model = ActorCritic(env, gamma=0.99, log_interval=100, max_episodes=10**6,
                       max_ep_length=30)
    model.train()


if __name__ == '__main__':
    main()

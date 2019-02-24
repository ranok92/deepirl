import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

import numpy as np

import gym
from rlmethods.b_actor_critic import ActorCritic


def main():
    env = gym.make('CartPole-v0')

    model = ActorCritic(env, gamma=0.99, log_interval=1, max_ep_length=200)
    model.train()


if __name__ == '__main__':
    main()

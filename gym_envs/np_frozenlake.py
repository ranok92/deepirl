'''
Wrapper around frozenlake-v0 that outputs one-hot numpy vectors instead of
integers as the state space.
'''
import pdb
from gym.envs.toy_text import frozen_lake
import numpy as np

import sys
sys.path.insert(0, '..')  # NOQA:E402:w
from utils import to_oh


class FrozenLakeNP(frozen_lake.FrozenLakeEnv):

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        return to_oh(state, self.nS)

    def step(self, *args, **kwargs):
        s, r, d, p = super().step(*args, **kwargs)
        s = to_oh(s, self.nS)

        return s, r, d, p

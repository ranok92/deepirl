"""A simple torch based gridworld."""

import numpy as np
from gym.spaces import Discrete


class IterableDiscrete(Discrete):
    """A version of gym's Discrete actions space that can be iterated over."""

    def __iter__(self):
        for i in range(0, self.n):
            yield i


class SimpleGridworld:
    """A simple numpy based gridworld."""

    def __init__(
            self,
            height,
            width,
            obstacles_map,
            player_pos,
            goal_pos,
    ):

        # size parameters
        self.height = height
        self.width = width

        # top left of grid is 0,0 as per image standards
        self.grid = np.array(height, width)

        # actions space mappings:
        # {0,1,2,3,4} = {left, up, down, right, stay}
        self.action_space = IterableDiscrete(5)
        self.action_dict = {
            0: np.array((-1, 0)),
            1: np.array((0, -1)),
            2: np.array((1, 0)),
            3: np.array((0, 1)),
            4: np.array((0, 0)),
        }

        self.obstacles = obstacles_map
        self.player_pos = player_pos
        self.goal_pos = goal_pos

    def reward_function(self, state, action, next_state):
        if self.goal_pos == next_state:
            return np.array([1])

        return np.array([0])

    def step(self, action):
        """Advance the gridworld player based on action.
        Returns (next_state, reward, done, False)

        :param action: Action peformed by player.
        """
        assert self.action_space.contains(action), "Invalid action!"

        action_vector = self.action_dict[action]
        state = self.player_pos

        next_state = state + action_vector
        next_state = next_state.clip([0, 0], [self.width, self.height])
        self.player_pos = next_state

        # reward function r(s_t, a_t, s_t+1)
        reward = reward_function(self, )

        done = (self.player_pos == self.goal_pos)

        return self.player_pos, reward, done, False

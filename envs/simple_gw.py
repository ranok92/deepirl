"""A simple torch based gridworld."""

import numpy as np
from gym.spaces import Discrete
import pdb


class IterableDiscrete(Discrete):
    """A version of gym's Discrete actions space that can be iterated over."""

    def __iter__(self):
        for i in range(0, self.n):
            yield i


class SimpleGridworld:
    """A simple numpy based gridworld."""

    def __init__(
            self,
            size,
            obstacles_map,
            goal_pos,
    ):
        """__init__

        :param size: a tuple of the form (rows, columns).
        :param obstacles_map: array-like of (row, column) coordinates of all
        obstacles.
        :param player_pos: (row, column) coordinate of player starting
        position.
        :param goal_pos: (row, column) coordinate of goal.
        """

        # top left of grid is 0,0 as per image standards
        self.grid = np.zeros(size)
        self.obstacles_map = obstacles_map

        # actions space mappings:
        # {0,1,2,3,4} = {up,left,down,right,stay}
        self.action_space = IterableDiscrete(5)
        self.action_dict = {
            0: np.array((-1, 0)),
            1: np.array((0, -1)),
            2: np.array((1, 0)),
            3: np.array((0, 1)),
            4: np.array((0, 0)),
        }

        self.obstacles = obstacles_map
        self.goal_pos = goal_pos
        self.player_pos = goal_pos

    def reset(self):
        # reset grid
        self.grid.fill(0)

        # fill obstacles (2 = obstacle)
        self.grid[self.obstacles_map.T[0], self.obstacles_map.T[1]] = 2

        # set goal
        assert self.grid[tuple(self.goal_pos)] != 2, "Goal is an obstacle."
        self.grid[tuple(self.goal_pos)] = 6

        # generate player location
        validity_condition = np.logical_or(self.grid != 2, self.grid != 6)
        valid_spots = np.argwhere(validity_condition)
        self.player_pos = valid_spots[np.random.choice(valid_spots.shape[0])]

        return self.player_pos

    def reward_function(self, state, action, next_state):
        """Generate a reward based on inputs.

        :param state: Current state (s_t)
        :param action: Action a_t taken at state s_t.
        :param next_state: State resulting from performing a_t at s_t.
        """
        if (self.goal_pos == next_state).all():
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
        next_state = next_state.clip(
            [0, 0],
            [self.grid.shape[0]-1, self.grid.shape[1]-1]
        )
        self.player_pos = next_state

        # reward function r(s_t, a_t, s_t+1)
        reward = self.reward_function(state, action, next_state)

        done = (self.player_pos == self.goal_pos).all()

        return self.player_pos, reward, done, False

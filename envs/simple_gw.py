"""A simple torch based gridworld."""

import numpy as np
import torch
from gym.spaces import Discrete
import pdb

# Define the default tensor type
torch.set_default_tensor_type(
    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
)


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

        return self.player_pos.astype('float32')

    def reward_function(self, state, action, next_state):
        """Generate a reward based on inputs.

        :param state: Current state (s_t)
        :param action: Action a_t taken at state s_t.
        :param next_state: State resulting from performing a_t at s_t.
        """
        goal_vector = self.goal_pos - self.player_pos
        movement_vector = next_state - state
        reward = np.sign(np.dot(goal_vector, movement_vector))
        return reward.astype('float32')

    def step(self, action):
        """Advance the gridworld player based on action.
        Returns (next_state, reward, done, False)

        :param action: Action peformed by player.
        """
        # convert pytorch tensor to int
        action = int(action.item())
        assert self.action_space.contains(action), "Invalid action!"

        action_vector = self.action_dict[action]
        state = self.player_pos

        next_state = state + action_vector
        next_state = next_state.clip(
            [0, 0],
            [self.grid.shape[0] - 1, self.grid.shape[1] - 1]
        )
        self.player_pos = next_state

        # reward function r(s_t, a_t, s_t+1)
        reward = self.reward_function(state, action, next_state)

        done = (self.player_pos == self.goal_pos).all()

        return self.player_pos.astype('float32'), reward, done, False


class TorchGridworld:
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
        self.grid = torch.zeros(size)
        self.obstacles_map = torch.tensor(obstacles_map)

        # actions space mappings:
        # {0,1,2,3,4} = {up,left,down,right,stay}
        self.action_space = IterableDiscrete(5)
        self.action_dict = {
            0: torch.tensor([-1, 0]),
            1: torch.tensor([0, -1]),
            2: torch.tensor([1, 0]),
            3: torch.tensor([0, 1]),
            4: torch.tensor([0, 0]),
        }

        self.goal_pos = torch.tensor(goal_pos)
        self.player_pos = torch.tensor(goal_pos)

        # pre-allocate min,max bounds
        self.min_pos = torch.tensor([0, 0])
        self.max_pos = torch.tensor([self.grid.shape[0], self.grid.shape[1]])

    def reset(self):
        # reset grid
        self.grid.fill_(0)

        # fill obstacles (2 = obstacle)
        self.grid[torch.t(self.obstacles_map)[0],
                  torch.t(self.obstacles_map)[1]] = 2

        # set goal
        assert self.grid[tuple(self.goal_pos)] != 2, "Goal is an obstacle."
        self.grid[tuple(self.goal_pos)] = 6

        # generate player location
        no_obstacle_mask = self.grid != 2
        no_goal_mask = self.grid != 6
        valid_spots = torch.nonzero(no_obstacle_mask | no_goal_mask)
        random_index = np.random.randint(0, valid_spots.shape[0])
        self.player_pos = valid_spots[random_index]

        return self.player_pos

    def reward_function(self, state, action, next_state):
        """Generate a reward based on inputs.

        :param state: Current state (s_t)
        :param action: Action a_t taken at state s_t.
        :param next_state: State resulting from performing a_t at s_t.
        """
        if torch.equal(self.goal_pos, next_state):
            return torch.tensor([1])

        return torch.tensor([0])

    def step(self, action):
        """Advance the gridworld player based on action.
        Returns (next_state, reward, done, False)

        :param action: Action peformed by player.
        """
        assert self.action_space.contains(action), "Invalid action!"

        with torch.no_grad():
            action_vector = self.action_dict[action]
            state = self.player_pos

            next_state = state + action_vector

            # clamp player pos inside gridworld
            next_state = torch.min(
                torch.max(next_state, self.min_pos), self.max_pos)

            self.player_pos = next_state

            # reward function r(s_t, a_t, s_t+1)
            reward = self.reward_function(state, action, next_state)

            done = bool((self.player_pos == self.goal_pos).all())

        return self.player_pos, reward, done, False

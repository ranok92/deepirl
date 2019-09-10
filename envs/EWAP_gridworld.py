"""Incorporate ETH Zurich's BIWI (EWAP) dataset into simple gridworld."""

import os
from pathlib import Path
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt

from .simple_gw import SimpleGridworld

# Grid constants
OBSTACLE = 2
GOAL = 6
PERSON = 9
ROBOT = 15


class EwapDataset:
    """
    Contains all relevant information for EWAP dataset.
    """

    def __init__(
            self,
            dataset_root='datasets/ewap_dataset',
            sequence='seq_hotel'
    ):
        dataset_path = Path(os.path.abspath(__file__)).parents[0]
        self.sequence_path = dataset_path / dataset_root / sequence

        # get obstacle map
        obs_map_path = self.sequence_path / 'hmap.png'
        self.obstacle_map = imread(str(obs_map_path.resolve()))
        self.obstacle_map *= OBSTACLE

        # get pedestrian position and velocity data (obsmat.txt)
        self.pedestrian_data = np.loadtxt(self.sequence_path / 'obsmat.txt')

        # get shift and scale amount for this sequence
        self.pos_shift = np.loadtxt(self.sequence_path / 'shift.txt')
        self.scale_factor = np.loadtxt(self.sequence_path / 'scale.txt')

        self.frame_id = 0
        self.processed_data = self.process_data()

    def scale(self, raw_data):
        scaled_data = raw_data.copy()
        scaled_data[:, 2:] *= self.scale_factor

        return scaled_data

    def shift(self, scaled_ped_data):
        shifted_ped_data = scaled_ped_data.copy()
        shifted_ped_data[:, 2] = shifted_ped_data[:, 2] - self.pos_shift[0]
        shifted_ped_data[:, 4] = shifted_ped_data[:, 4] - self.pos_shift[1]

        return shifted_ped_data

    def discretize(self, scaled_shifted_data):
        discrete_data = np.round(scaled_shifted_data).astype(np.int64)

        return discrete_data

    def normalize_frames(self, unnormalied_data):
        """
        Normalizers the data frames, so first frame is integer 0.
        """
        normalized_data = unnormalied_data.copy()
        normalized_data[:, 0] -= np.min(normalized_data[:, 0])
        normalized_data[:, 0] = normalized_data[:, 0] / 10

        return normalized_data

    def process_data(self):
        """
        scales, shifts, and discretizes input data according to parameters
        generated by map2world.py script. normalizes frame numbers.
        """

        scaled_data = self.scale(self.pedestrian_data)
        shifted_data = self.shift(scaled_data)
        discrete_data = self.discretize(shifted_data)

        # normalize frames
        processed_data = self.normalize_frames(discrete_data)

        return processed_data

    def pad_dataset(self, pad_amount):
        """
        Pad dataset with obstacle pixels to ensure 2*vision_radius+1 squares
        are possible for feature extractor.
        """
        self.obstacle_map = np.pad(
            self.obstacle_map,
            pad_amount + 1,  # for when agent runs into edges
            mode='constant',
            constant_values=OBSTACLE
        )

        # shift dataset to accomodate padding
        self.processed_data[:, 2] += pad_amount
        self.processed_data[:, 4] += pad_amount

    def pedestrian_goal(self, pedestrian_id):
        """Return goal of pedestrian with the given ID.

        :param pedestrian_id: ID of pedestrian.
        """
        pedestrian_mask = (self.processed_data[:, 1] == pedestrian_id)
        pedestrian_traj = self.processed_data[pedestrian_mask]

        return (pedestrian_traj[-1, 2], pedestrian_traj[-1, 4])


class EwapGridworld(SimpleGridworld):
    """
    Env that incorporates ETHZ BIWI (EWAP) dataset.
    make sure the correct directory sturcture exists in order to run, i.e.

    /envs/datasets/ewap_dataset/seq_eth
    /envs/datasets/ewap_dataset/seq_hotel

    folders exist. Additionally, run script map2world.py for both sequences,
    instruction found in the script at /envs/datasets/ewap_dataset/map2world.py
    """

    def __init__(
            self,
            ped_id,
            sequence='seq_hotel',
            dataset_root='datasets/ewap_dataset',
            person_thickness=2,
            vision_radius=4,
    ):
        """
        Initialize EWAP gridworld. Make sure all files in the correct
        directories as specified above!

        :param ped_id: ID of pedestrian whose goal we adopt.
        :param sequence: Sequence to base world on. example: 'seq_hotel'
        :param dataset_root: path to root of dataset directory.
        :param person_thickness: The dimensions of the nxn boxes that will
        represent people.
        """

        self.dataset = EwapDataset(
            sequence=sequence,
            dataset_root=dataset_root
        )

        self.vision_radius = vision_radius
        self.dataset.pad_dataset(vision_radius)

        self.person_thickness = person_thickness

        obstacle_array = np.where(self.dataset.obstacle_map == OBSTACLE)
        obstacle_array = np.array(obstacle_array).T

        self.goals = self.dataset.pedestrian_goal(ped_id)

        super().__init__(
            self.dataset.obstacle_map.shape,
            obstacle_array,
            self.goals
        )

        self.adopt_goal(ped_id)

        # seperate for people position incase we want to do processing.
        self.person_map = self.grid.copy()

        self.person_map.fill(0)
        self.png_number = 0

    def thicken(self, grid, target_thickness):
        """
        thicken pixel by specified target_thickness parameter in supplied grid.

        :param target_thickness: thickness amount.
        :param grid: grid in which pixels reside.
        """

        row_ind, col_ind = np.where(grid != 0)
        thick = grid.copy()

        assert row_ind.size == col_ind.size

        for i in range(row_ind.size):
            row = row_ind[i]
            col = col_ind[i]
            thick[
                row - target_thickness:row + target_thickness + 1,
                col - target_thickness:col + target_thickness + 1
            ] = thick[row, col]

        return thick

    def populate_person_map(self, frame_num):
        """Populates the person map based on input frame_num, which is the
        frame id of EWAP database's footage.

        :param frame_num: frame to get positions from.
        """
        # clear person map
        self.person_map.fill(0)

        # Get data for current frame of simulation.
        frame_pedestrians = self.dataset.processed_data[
            self.dataset.processed_data[:, 0] == frame_num
        ]

        self.person_map[frame_pedestrians[:, 2],
                        frame_pedestrians[:, 4]] = PERSON
        self.person_map = self.thicken(self.person_map, self.person_thickness)

    def adopt_goal(self, pedestrian_id):
        self.goals = self.dataset.pedestrian_goal(pedestrian_id)
        self.goal_grid[self.goals[0], self.goals[1]] = GOAL
        self.goal_grid = self.thicken(self.goal_grid, self.person_thickness*5)

    def layer_map(self):
        """
        Layer the gridworld map in the proper order:
            static obstacles > pedestrians > goal position
        """
        self.grid.fill(0)
        self.grid += self.obstacle_grid

        # Add pedestrians
        self.populate_person_map(self.step_number)
        self.grid += self.person_map

    def reset(self):
        """
        reset gridworld to initial positon, with all trajectories starting
        again at first frame.
        """

        super().reset()
        assert self.step_number == 0, 'Step number non-zero after reset!'

        self.populate_person_map(self.step_number)
        self.grid += self.person_map

        return self.state_extractor().astype('float32')

    def reward_function(self, state, action, next_state):
        reward = super().reward_function(state, action, next_state)

        if self.grid[tuple(self.player_pos)] == PERSON:
            reward += -2.0

        # determine if moving closer to goal
        current_distance = np.sum(np.abs(state[-4:-2] - state[-2:]))
        next_distance = np.sum(np.abs(next_state[-4:-2] - next_state[-2:]))

        reward += 0.0001 * (next_distance < current_distance).astype(int)
        reward -= 0.0001 * (next_distance > current_distance).astype(int)

        return reward

    def state_extractor(self):
        """
        Extract state for CURRENT internal state of gridworld.
        """

        surroundings = self.grid[
            self.player_pos[0] - self.vision_radius: self.player_pos[0] + self.vision_radius + 1,
            self.player_pos[1] - self.vision_radius: self.player_pos[1] + self.vision_radius + 1,
        ]

        try:
            assert surroundings.shape == (
                2 * self.vision_radius + 1, 2 * self.vision_radius + 1)
        except:
            breakpoint()

        # state vector is surroundings concatenated with player pos and goal
        # pos, hence the +4 size
        state = np.zeros(surroundings.size + 4)
        state[0: surroundings.size] = surroundings.flatten()
        state[-4:-2] = self.player_pos
        state[-2:] = np.array(self.goal_pos)

        return state

    def step(self, action):
        """
        Advance the gridworld player based on action.
        'done' is set when environment reaches goal, hits obstacle, or exceeds
        max step number, which is heuristically set to length*height of
        gridworld.

        :param action: Action peformed by player.
        :return (next_state, reward, done, False)
        """
        assert self.action_space.contains(action), "Invalid action!"

        # extract current
        state = self.state_extractor().astype('float32')

        # advance player based on action
        action_vector = self.action_dict[action]
        next_pos = self.player_pos + action_vector
        self.step_number += 1

        # advance simulation
        self.layer_map()

        # extract next state
        self.player_pos = next_pos
        next_state = self.state_extractor().astype('float32')

        # reward function r(s_t, a_t, s_t+1)
        reward = self.reward_function(state, action, next_state)

        goal_reached = (self.goal_grid[tuple(self.player_pos)] == GOAL)
        obstacle_hit = (self.grid[tuple(self.player_pos)] == OBSTACLE)
        person_hit = (self.grid[tuple(self.player_pos)] == PERSON)
        max_steps_elapsed = self.step_number > self.grid.size

        done = goal_reached or obstacle_hit or max_steps_elapsed or person_hit

        return next_state, reward, done, False

    def dump_png(self, path='./png_dumps/'):
        impath = Path(path)
        image_name = str(self.png_number) + '.png'
        to_save = self.grid + self.goal_grid + self.obstacle_grid
        to_save[tuple(self.player_pos)] = ROBOT
        plt.imsave(str((impath / image_name).resolve()), to_save)
        self.png_number += 1

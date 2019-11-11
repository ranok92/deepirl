from featureExtractor.gridworld_featureExtractor import SocialNav, LocalGlobal, FrontBackSideSimple
import copy
from envs.gridworld import GridWorld
from itertools import count
from alternateController.potential_field_controller import PotentialFieldController as PFController
from envs.drone_env_utils import InformationCollector
from featureExtractor.drone_feature_extractor import DroneFeatureSAM1, DroneFeatureMinimal, DroneFeatureOccup, DroneFeatureRisk, DroneFeatureRisk_v2
from envs.gridworld_clockless import MockActionspace, MockSpec
import numpy as np
import torch
import time
import pdb
import sys
import sys
import math
import os
sys.path.insert(0, '..')


import utils  # NOQA: E402
with utils.HiddenPrints():
    import pygame
    import pygame.freetype


class Pedestrian():

    def __init__(self,
                 idval=None,
                 pos=None,
                 speed=None,
                 orientation=None
                 ):
        self.id = idval
        self.position = pos
        self.speed = speed
        self.orientation = orientation


class GridWorldDrone(GridWorld):
    def __init__(
            self,
            seed=7,
            rows=10,
            cols=10,
            width=10,
            goal_state=None,
            obstacles=None,
            display=True,
            is_onehot=True,
            is_random=False,
            step_reward=0.001,
            step_wrapper=utils.identity_wrapper,
            reset_wrapper=utils.identity_wrapper,
            show_trail=False,
            show_orientation=False,
            annotation_file=None,
            subject=None,
            obs_width=10,
            step_size=10,
            agent_width=10,
            show_comparison=False,
            tick_speed=30,
            replace_subject=False,
            external_control=True,
            consider_heading=False,
            variable_speed=False
    ):
        super().__init__(seed=seed,
                         rows=rows,
                         cols=cols,
                         width=width,
                         goal_state=goal_state,
                         obstacles=obstacles,
                         display=display,
                         is_onehot=is_onehot,
                         is_random=is_random,
                         step_reward=step_reward,
                         step_wrapper=step_wrapper,
                         reset_wrapper=reset_wrapper,
                         show_trail=show_trail,
                         obs_width=obs_width,
                         consider_heading=consider_heading,
                         agent_width=agent_width,
                         step_size=step_size
                         )
        if display:

            self.gameDisplay = pygame.display.set_mode((self.cols, self.rows))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.env_font = pygame.font.SysFont('Comic Sans MS', 20)
            self.tickSpeed = tick_speed

        self.show_comparison = show_comparison

        self.ghost = None
        self.ghost_state = None
        self.ghost_state_history = []
        self.ghost_color = (140, 0, 200)

        # the file from which the video information will be used
        self.annotation_file = annotation_file
        self.annotation_dict = {}
        self.pedestrian_dict = {}
        self.current_frame = 0
        self.final_frame = -1
        self.initial_frame = 999999999999  # a large number

        self.subject = subject
        self.cur_ped = None

        self.max_obstacles = None
        self.agent_action_flag = False
        self.obstacle_width = obs_width
        self.step_size = step_size
        self.show_trail = show_trail
        self.annotation_list = []
        self.skip_list = []  # dont consider these pedestrians as obstacles

        ############# this comes with the change in the action space##########
        self.actionArray = [np.asarray([-1, 0]), np.asarray([-1, 1]),
                            np.asarray([0, 1]), np.asarray([1, 1]),
                            np.asarray([1, 0]), np.asarray([1, -1]),
                            np.asarray([0, -1]), np.asarray([-1, -1]), np.asarray([0, 0])]

        self.speed_array = np.array([1, 0.75, 0.5, 0.25, 0])

        self.action_dict = {}
        self.prev_action = 8
        for i in range(len(self.actionArray)):
            self.action_dict[np.array2string(self.actionArray[i])] = i

        self.action_space = MockActionspace(len(self.actionArray))
        self.step_reward = step_reward
        self.external_control = external_control
        self.replace_subject = replace_subject

        self.show_orientation = show_orientation

    def render(self):
        # render board
        self.clock.tick(self.tickSpeed)
        font = pygame.freetype.Font(None, 15)
        self.gameDisplay.fill(self.white, [0, 0, self.cols, self.rows])
        # render obstacles
        if self.obstacles is not None:
            for obs in self.obstacles:
                pygame.draw.rect(self.gameDisplay,
                                 self.red,
                                 [obs['position'][1] - (self.obs_width / 2),
                                  obs['position'][0] - (self.obs_width / 2),
                                     self.obs_width,
                                     self.obs_width])
                font.render_to(self.gameDisplay,
                               (obs['position'][1] - (self.obs_width / 2) - 5,
                                obs['position'][0] - (self.obs_width / 2)),
                               obs['id'], fgcolor=(0, 0, 0))
                if self.show_orientation:
                    if obs['orientation'] is not None:
                        pygame.draw.line(
                            self.gameDisplay, self.black, [
                                obs['position'][1], obs['position'][0]], [
                                obs['position'][1] + obs['orientation'][1] * 10, obs['position'][0] + obs['orientation'][0] * 10], 2)
        # render goal
        if self.goal_state is not None:
            pygame.draw.rect(self.gameDisplay,
                             self.green,
                             [self.goal_state[1] - (self.cellWidth / 2),
                              self.goal_state[0] - (self.cellWidth / 2),
                                 self.cellWidth,
                                 self.cellWidth])
        # render agent
        if self.agent_state is not None:
            pygame.draw.rect(self.gameDisplay,
                             self.black,
                             [self.agent_state['position'][1] - (self.agent_width / 2),
                              self.agent_state['position'][0] - (self.agent_width / 2),
                                 self.agent_width,
                                 self.agent_width])

            if self.show_orientation:
                if self.agent_state['orientation'] is not None:
                    pygame.draw.line(
                        self.gameDisplay,
                        self.black,
                        [
                            self.agent_state['position'][1],
                            self.agent_state['position'][0]],
                        [
                            self.agent_state['position'][1] +
                            self.agent_state['orientation'][1] *
                            10,
                            self.agent_state['position'][0] +
                            self.agent_state['orientation'][0] *
                            10],
                        2)

        if self.show_trail:
            self.draw_trajectory(self.pos_history, self.black)

        pygame.display.update()

    def step(self, action):
        """ Step forward in simulation where agent takes some action.
        
        :param action: Action taken by agent
        :type action: int
        :return: (next_state, reward, done, info)
        :rtype: (np.array, Float, Boolean, _)
        """
        # TODO: Implement this!
        raise(NotImplementedError)

    def calculate_reward(self, cur_action):
        # TODO: Implement reward!
        raise(NotImplementedError)

    def reset(self):
        # TODO: Implement a reset!
        raise(NotImplementedError)

    def close_game(self):
        pygame.quit()

    def draw_arrow(self, base_position, next_position, color):
        #base_position = (row,col)
        if np.linalg.norm(
                base_position - next_position) <= self.step_size * math.sqrt(2):
            # draw the stalk
            arrow_width = self.cellWidth * .1  # in pixels
            base_pos_pixel = (base_position + .5)
            next_pos_pixel = (next_position + .5)
            # pdb.set_trace()

            # draw the head
            ref_pos = base_pos_pixel + (next_pos_pixel - base_pos_pixel) * .35
            arrow_length = 0.7
            arrow_base = base_pos_pixel
            arrow_end = base_pos_pixel + \
                (next_pos_pixel - base_pos_pixel) * arrow_length

            pygame.draw.line(
                self.gameDisplay,
                color,
                (arrow_base[1],
                 arrow_base[0]),
                (arrow_end[1],
                 arrow_end[0]),
                2)

    def draw_trajectory(self, trajectory=[], color=None):
        # TODO: Parametrized hard-coded stuff below.
        arrow_length = 1
        arrow_head_width = 1
        arrow_width = .1
        # denotes the start and end positions of the trajectory
        rad = int(self.cellWidth * .4)
        start_pos = (trajectory[0]['position'] + .5) * self.cellWidth
        end_pos = (trajectory[-1]['position'] + 0.5) * self.cellWidth

        pygame.draw.circle(self.gameDisplay, (0, 255, 0),
                           (int(start_pos[1]), int(start_pos[0])),
                           rad)

        pygame.draw.circle(self.gameDisplay, (0, 0, 255),
                           (int(end_pos[1]), int(end_pos[0])),
                           rad)

        for count in range(len(trajectory) - 1):
            # pygame.draw.lines(self.gameDisplay,color[counter],False,trajectory_run)
            self.draw_arrow(trajectory[count]['position'],
                            trajectory[count + 1]['position'], color)

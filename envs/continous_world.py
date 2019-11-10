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

    # the numbering starts from 0,0 from topleft corner and goes down and right
    # the obstacles should be a list of 2 dim numpy array stating the position of the
    # obstacle
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

    def get_state_from_frame(self, frame_info):
        '''
        For stanford dataset
        '''
        self.obstacles = []
        for element in frame_info:

            if element[6] != str(1):  # they are visible
                if int(element[0]) != self.cur_ped:

                    left = int(element[1])
                    top = int(element[2])
                    width = int(element[3]) - left
                    height = int(element[4]) - top
                    self.obstacles.append(
                        np.array([int(top + (height / 2)), int(left + (width / 2))]))

                else:

                    left = int(element[1])
                    top = int(element[2])
                    width = int(element[3]) - left
                    height = int(element[4]) - top
                    self.agent_state = np.array(
                        [int(top + (height / 2)), int(left + (width / 2))])

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

        if self.ghost_state is not None:
            pygame.draw.rect(self.gameDisplay,
                             self.ghost_color,
                             [self.ghost_state['position'][1] - (self.agent_width / 2),
                              self.ghost_state['position'][0] - (self.agent_width / 2),
                                 self.agent_width,
                                 self.agent_width])

        if self.show_trail:
            self.draw_trajectory(self.pos_history, self.black)

            if self.ghost:
                self.draw_trajectory(
                    self.ghost_state_history, self.ghost_color)

        pygame.display.update()
        return 0

    def step(self, action=None):
        '''
        if external_control: t
            the state of the agent is updated based on the current action.
        else:
            the state of the agent is updated based on the information from the frames

        the rest of the actions, like calculating reward and checking if the episode is done remains as usual.
        '''
        self.current_frame += 1

        if str(self.current_frame) in self.annotation_dict.keys():
            self.get_state_from_frame_universal(
                self.annotation_dict[str(self.current_frame)])

        if self.external_control:

            if not self.release_control:

                if action is not None:
                    if isinstance(action, int):

                        if action != 8 and self.consider_heading:
                            action = (self.cur_heading_dir + action) % 8
                            self.cur_heading_dir = action
                        # self.heading_dir_history.append(self.cur_heading_dir)
                        #self.cur_heading_dir = action
                        prev_position = self.agent_state['position']
                        self.agent_state['position'] = np.maximum(
                            np.minimum(
                                self.agent_state['position'] +
                                self.step_size *
                                self.actionArray[action],
                                self.upper_limit_agent),
                            self.lower_limit_agent)

                        self.agent_state['orientation'] = self.agent_state['position'] - prev_position
                        self.agent_state['speed'] = np.linalg.norm(
                            self.agent_state['orientation'])

                    else:
                        # if the action is a torch
                        # check if it the tensor has a single value
                        if len(action.shape) == 1 and a.shape[0] == 1:
                            if isinstance(action.item(), int):
                                prev_position = self.agent_state['position']

                                if action != 8 and self.consider_heading:
                                    action = (
                                        self.cur_heading_dir + action) % 8
                                    self.cur_heading_dir = action
                                # self.heading_dir_history.append(self.cur_heading_dir)
                                #self.cur_heading_dir = action

                                self.agent_state['position'] = np.maximum(
                                    np.minimum(
                                        self.agent_state['position'] +
                                        self.step_size *
                                        self.actionArray[action],
                                        self.upper_limit_agent),
                                    self.lower_limit_agent)

                                self.agent_state['orientation'] = self.agent_state['position'] - prev_position
                                self.agent_state['speed'] = np.linalg.norm(
                                    self.agent_state['orientation'])
                    #print("Agent :",self.agent_state)

            # if not np.array_equal(self.pos_history[-1],self.agent_state):
            self.heading_dir_history .append(self.cur_heading_dir)

            self.pos_history.append(copy.deepcopy(self.agent_state))

            if self.ghost:
                self.ghost_state_history.append(
                    copy.deepcopy(self.ghost_state))

        # calculate the reward and completion condition
        reward, done = self.calculate_reward(action)
        self.prev_action = action

        # if you are done ie hit an obstacle or the goal
        # you leave control of the agent and you are forced to
        # suffer/enjoy the consequences of your actions for the
        # rest of your miserable/awesome life

        if self.display:
            self.render()

        # step should return fourth element 'info'
        if self.is_onehot:
            self.state = self.onehotrep()
        else:
            # just update the position of the agent
            # the rest of the information remains the same

            # added new
            if not self.release_control:
                self.state['agent_state'] = copy.deepcopy(self.agent_state)
                if action != 8:
                    self.state['agent_head_dir'] = action

        if self.is_onehot:

            self.state, reward, done, _ = self.step_wrapper(
                self.state,
                reward,
                done,
                None
            )

        if self.external_control:
            if done:
                self.release_control = True

        return self.state, reward, done, None

    def calculate_reward(self, cur_action):

        hit = False
        done = False

        if self.obstacles is not None:
            for obs in self.obstacles:
                if self.check_overlap(
                        self.agent_state['position'],
                        obs['position'],
                        self.obs_width,
                        self.buffer_from_obs):
                    hit = True

        if (hit):
            reward = -1
            done = True

        elif self.check_overlap(self.agent_state['position'], self.goal_state, self.cellWidth, 0):
            reward = 1
            done = True

        else:

            newdist = np.linalg.norm(
                self.agent_state['position'] - self.goal_state, 1)

            reward = (self.distanceFromgoal - newdist) * self.step_reward

            self.distanceFromgoal = newdist

        if cur_action is not None:

            energy_spent = - \
                np.sum(
                    np.square(self.actionArray[cur_action] - self.actionArray[self.prev_action]))

            reward += energy_spent * self.step_reward * 1

        # pdb.set_trace()
        return reward, done

    def reset(self):
        '''
        Resets the environment, starting the obstacles from the start.
        If subject is specified, then the initial frame and final frame is set
        to the time frame when the subject is in the scene.

        If no subject is specified then the initial frame is set to the overall
        initial frame and goes till the last frame available in the annotation file.

        Also, the agent and goal positions are initialized at random.

        Pro tip: Use this function while training the agent.
        '''
        # pygame.image.save(self.gameDisplay,'traced_trajectories.png')
        #########for debugging purposes###########
        if self.replace_subject:

            return self.reset_and_replace()

        else:

            #self.skip_list = [i for i in range(len(self.pedestrian_dict.keys()))]

            self.current_frame = self.initial_frame
            self.pos_history = []
            self.ghost_state_history = []
            # if this flag is true, the position of the obstacles and the goal
            # change with each reset
            dist_g = self.goal_spawn_clearance

            if self.annotation_file:
                self.get_state_from_frame_universal(
                    self.annotation_dict[str(self.current_frame)])

            num_obs = len(self.obstacles)

            # placing the obstacles

            # only for the goal and the agent when the subject is not specified
            # speicfically.

            if self.cur_ped is None:

                # placing the goal
                while True:
                    flag = False
                    self.goal_state = np.asarray(
                        [
                            np.random.randint(
                                self.lower_limit_goal[0],
                                self.upper_limit_goal[0]),
                            np.random.randint(
                                self.lower_limit_goal[1],
                                self.upper_limit_goal[1])])

                    for i in range(num_obs):
                        if np.linalg.norm(
                                self.obstacles[i]['position'] -
                                self.goal_state) < dist_g:

                            flag = True
                    if not flag:
                        break

                # placing the agent
                dist = self.agent_spawn_clearance
                while True:
                    flag = False
                    # pdb.set_trace()
                    self.agent_state['position'] = np.asarray(
                        [
                            np.random.randint(
                                self.lower_limit_agent[0],
                                self.upper_limit_agent[0]),
                            np.random.randint(
                                self.lower_limit_agent[1],
                                self.upper_limit_agent[1])])

                    for i in range(num_obs):
                        if np.linalg.norm(
                                self.obstacles[i]['position'] -
                                self.agent_state['position']) < dist:
                            flag = True

                    if not flag:
                        break

            self.release_control = False
            if self.is_onehot:
                self.state = self.onehotrep()
            else:
                self.state = {}
                self.state['agent_state'] = copy.deepcopy(self.agent_state)
                self.state['agent_head_dir'] = 0  # starts heading towards top
                self.state['goal_state'] = self.goal_state

                self.state['release_control'] = self.release_control
                # if self.obstacles is not None:
                self.state['obstacles'] = self.obstacles

            self.pos_history.append(copy.deepcopy(self.agent_state))
            if self.ghost:
                self.ghost_state_history.append(
                    copy.deepcopy(self.ghost_state))

            self.distanceFromgoal = np.linalg.norm(
                self.agent_state['position'] - self.goal_state, 1)
            self.cur_heading_dir = 0
            self.heading_dir_history = []
            self.heading_dir_history.append(self.cur_heading_dir)

            pygame.display.set_caption('Your friendly grid environment')
            if self.display:
                self.render()

            if self.is_onehot:
                self.state = self.reset_wrapper(self.state)
            return self.state

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

"""A continous 2D top down world with continous action space."""

import math

import numpy as np

import utils  # NOQA: E402
from .gridworld import GridWorld

with utils.HiddenPrints():
    import pygame
    import pygame.freetype


class ContinousWorld(GridWorld):
    """ Class representing a continous 2D world to navigate in. """

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
        consider_heading=False,
    ):
        super().__init__(
            seed=seed,
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
            step_size=step_size,
        )
        if display:
            self.gameDisplay = pygame.display.set_mode((self.cols, self.rows))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.env_font = pygame.font.SysFont("Comic Sans MS", 20)
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
        self.skip_list = []  # dont consider these pedestrians as obstacles

        self.step_reward = step_reward
        self.show_orientation = show_orientation

    def render(self):
        # render board
        self.clock.tick(self.tickSpeed)
        font = pygame.freetype.Font(None, 15)
        self.gameDisplay.fill(self.white, [0, 0, self.cols, self.rows])
        # render obstacles
        if self.obstacles is not None:
            for obs in self.obstacles:
                pygame.draw.rect(
                    self.gameDisplay,
                    self.red,
                    [
                        obs["position"][1] - (self.obs_width / 2),
                        obs["position"][0] - (self.obs_width / 2),
                        self.obs_width,
                        self.obs_width,
                    ],
                )
                font.render_to(
                    self.gameDisplay,
                    (
                        obs["position"][1] - (self.obs_width / 2) - 5,
                        obs["position"][0] - (self.obs_width / 2),
                    ),
                    obs["id"],
                    fgcolor=(0, 0, 0),
                )
                if self.show_orientation:
                    if obs["orientation"] is not None:
                        pygame.draw.line(
                            self.gameDisplay,
                            self.black,
                            [obs["position"][1], obs["position"][0]],
                            [
                                obs["position"][1] + obs["orientation"][1] * 10,
                                obs["position"][0] + obs["orientation"][0] * 10,
                            ],
                            2,
                        )
        # render goal
        if self.goal_state is not None:
            pygame.draw.rect(
                self.gameDisplay,
                self.green,
                [
                    self.goal_state[1] - (self.cellWidth / 2),
                    self.goal_state[0] - (self.cellWidth / 2),
                    self.cellWidth,
                    self.cellWidth,
                ],
            )
        # render agent
        if self.agent_state is not None:
            pygame.draw.rect(
                self.gameDisplay,
                self.black,
                [
                    self.agent_state["position"][1] - (self.agent_width / 2),
                    self.agent_state["position"][0] - (self.agent_width / 2),
                    self.agent_width,
                    self.agent_width,
                ],
            )

            if self.show_orientation:
                if self.agent_state["orientation"] is not None:
                    pygame.draw.line(
                        self.gameDisplay,
                        self.black,
                        [
                            self.agent_state["position"][1],
                            self.agent_state["position"][0],
                        ],
                        [
                            self.agent_state["position"][1]
                            + self.agent_state["orientation"][1] * 10,
                            self.agent_state["position"][0]
                            + self.agent_state["orientation"][0] * 10,
                        ],
                        2,
                    )

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
        raise NotImplementedError

    def calculate_reward(self, cur_action):
        # TODO: Implement reward!
        raise NotImplementedError

    def reset(self):
        # TODO: Implement a reset!
        raise NotImplementedError

    def close_game(self):
        pygame.quit()

    def draw_arrow(self, base_position, next_position, color):
        # base_position = (row,col)
        if np.linalg.norm(
            base_position - next_position
        ) <= self.step_size * math.sqrt(2):
            # draw the stalk
            base_pos_pixel = base_position + 0.5
            next_pos_pixel = next_position + 0.5
            # pdb.set_trace()

            # draw the head
            arrow_length = 0.7
            arrow_base = base_pos_pixel
            arrow_end = (
                base_pos_pixel
                + (next_pos_pixel - base_pos_pixel) * arrow_length
            )

            pygame.draw.line(
                self.gameDisplay,
                color,
                (arrow_base[1], arrow_base[0]),
                (arrow_end[1], arrow_end[0]),
                2,
            )

    def draw_trajectory(self, trajectory=[], color=None):
        # TODO: Parametrized hard-coded stuff below.
        # denotes the start and end positions of the trajectory
        rad = int(self.cellWidth * 0.4)
        start_pos = (trajectory[0]["position"] + 0.5) * self.cellWidth
        end_pos = (trajectory[-1]["position"] + 0.5) * self.cellWidth

        pygame.draw.circle(
            self.gameDisplay,
            (0, 255, 0),
            (int(start_pos[1]), int(start_pos[0])),
            rad,
        )

        pygame.draw.circle(
            self.gameDisplay,
            (0, 0, 255),
            (int(end_pos[1]), int(end_pos[0])),
            rad,
        )

        for count in range(len(trajectory) - 1):
            self.draw_arrow(
                trajectory[count]["position"],
                trajectory[count + 1]["position"],
                color,
            )

"""Experiments with SAC+LSTM for EWAP env."""
import sys
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
sys.path.insert(0, '..')  # NOQA: E402

from rlmethods.soft_ac_pi import SoftActorCritic
from rlmethods.rlutils import ContigousReplayBuffer
from envs.EWAP_gridworld import EwapGridworld
from neural_nets.base_network import BaseNN

import gym

parser = ArgumentParser()
parser.add_argument('replay_buffer_size', type=int)
parser.add_argument('replay_buffer_sample_size', type=int)
parser.add_argument('--log-alpha', type=float, default=-2.995)
parser.add_argument('--entropy-target', type=float, default=1.0)
parser.add_argument('--max-episodes', type=int, default=10**4)
parser.add_argument('--play-interval', type=int, default=100)
parser.add_argument('--render', action='store_true')

args = parser.parse_args()


def conv_out_size(h_in, kernel_side, stride):
    return ((h_in - (kernel_side-1) - 1) // stride[0]) + 1


class ConvQNet(BaseNN):
    def __init__(
            self,
            state_length,
            action_length,
            hidden_layer_width,
            map_side,
            kernel_shape=(3, 3),
            pool_shape=(3, 3)
    ):
        super().__init__()

        # env information
        self.state_length = state_length
        self.map_side = map_side

        # convolutional layers
        self.pool = nn.MaxPool2d(pool_shape)

        self.conv1 = nn.Conv2d(3, 8, (8, 8), 4)
        self.conv2 = nn.Conv2d(8, 32, (4, 4), 2)
        self.conv3 = nn.Conv2d(32, 32, (2, 2), 1)

        # calculate length of vector input into fully connected layers
        reduced_side = conv_out_size(
            self.map_side,
            self.conv1.kernel_size[1],
            self.conv1.stride
        )
        reduced_side = conv_out_size(
            reduced_side,
            self.conv2.kernel_size[1],
            self.conv2.stride
        )
        reduced_side = conv_out_size(
            reduced_side,
            self.conv3.kernel_size[1],
            self.conv3.stride
        )
        fc_in_len = 6 + self.conv3.out_channels*(reduced_side**2)

        # create fully connected layers based on above
        self.fc1 = nn.Linear(fc_in_len, hidden_layer_width)
        self.fc2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.out_layer = nn.Linear(hidden_layer_width, action_length)

    def forward(self, state):
        # split statespace into respective maps based on each side of the
        # statespace maps (squares with sides = self.map_side)
        state_ = state.reshape(-1, state.shape[-1])

        in_obs = state_[:, :self.map_side ** 2]
        in_obs = in_obs.reshape(-1, self.map_side, self.map_side)

        in_persons = state_[:, self.map_side ** 2:2 * self.map_side ** 2]
        in_persons = in_persons.reshape(-1, self.map_side, self.map_side)

        in_goals = state_[:, 2 * self.map_side ** 2:3 * self.map_side ** 2]
        in_goals = in_goals.reshape(-1, self.map_side, self.map_side)

        in_surround = torch.stack((in_obs, in_persons, in_goals), dim=1)

        # convolutional processsing
        out_surround = F.relu(self.conv1(in_surround))
        out_surround = F.relu(self.conv2(out_surround))
        out_surround = F.relu(self.conv3(out_surround))

        # fully connected
        x = torch.cat(
            (
                out_surround.flatten(start_dim=1),
                state_[:, -6:]
            ),
            dim=1
        )

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out_layer(x)

        return x


class ConvPiNet(ConvQNet):

    def forward(self, state):
        x = super().forward(state)
        x = F.softmax(x, dim=-1)

        return x

    def action_distribution(self, state):
        probs = self.__call__(state)
        dist = Categorical(probs)

        return dist


def main():
    tbx_writer = SummaryWriter(comment='_alpha_' + str(args.log_alpha))

    env = EwapGridworld(ped_id=1, render=args.render, vision_radius=50,
                        person_thickness=1)

    state_size = env.reset().shape[0]
    map_side = (1 + env.vision_radius * 2)

    conv_q_net = ConvQNet(
        state_size,
        env.action_space.n,
        1024,
        map_side,
        kernel_shape=(3, 3)
    )

    conv_policy_net = ConvPiNet(
        state_size,
        env.action_space.n,
        1024,
        map_side,
        kernel_shape=(3, 3)
    )

    replay_buffer = ContigousReplayBuffer(args.replay_buffer_size)

    soft_ac = SoftActorCritic(
        env,
        replay_buffer=replay_buffer,
        buffer_sample_size=args.replay_buffer_sample_size,
        tbx_writer=tbx_writer,
        tau=0.005,
        log_alpha=args.log_alpha,
        entropy_tuning=True,
        entropy_target=args.entropy_target,
        q_net=conv_q_net,
        policy_net=conv_policy_net
    )

    soft_ac.train(args.max_episodes, args.play_interval)


if __name__ == "__main__":
    main()

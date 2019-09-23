"""Script for Q-based Soft Actor critic on the EWAP(BIWI) dataset."""
import sys
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
sys.path.insert(0, '..')  # NOQA: E402

from rlmethods.soft_ac import SoftActorCritic
from envs.EWAP_gridworld import EwapGridworld
from neural_nets.base_network import BaseNN


parser = ArgumentParser()
parser.add_argument('replay_buffer_size', type=int)
parser.add_argument('replay_buffer_sample_size', type=int)
parser.add_argument('--log-alpha', type=float, default=-2.995)
parser.add_argument('--entropy-target', type=float, default=1.0)
parser.add_argument('--max-episodes', type=int, default=10**4)
parser.add_argument('--play-interval', type=int, default=100)
parser.add_argument('--render', action='store_true')

args = parser.parse_args()


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
        """__init__

        :param 3:
        :param 3:
        """
        super().__init__()

        # env information
        self.state_length = state_length
        self.map_side = map_side

        # convolutional layers
        padding = (kernel_shape[0] - 1) // 2
        self.conv1_obstacles = nn.Conv2d(1, 1, kernel_shape, padding=padding)
        self.conv1_persons = nn.Conv2d(1, 1, kernel_shape, padding=padding)
        self.conv1_goals = nn.Conv2d(1, 1, kernel_shape, padding=padding)
        self.conv2_obstacles = nn.Conv2d(1, 1, kernel_shape, padding=padding)
        self.conv2_persons = nn.Conv2d(1, 1, kernel_shape, padding=padding)
        self.conv2_goals = nn.Conv2d(1, 1, kernel_shape, padding=padding)

        self.pool = nn.MaxPool2d(pool_shape)

        # calculate length of vector input into fully connected layers
        pool_reduction_factor = pool_shape[0] * pool_shape[1]
        conv2_in_len = (map_side**2) // pool_reduction_factor
        fc_in_len = 4 + (3 * (conv2_in_len // pool_reduction_factor))

        # create fully connected layers based on above
        self.fc1 = nn.Linear(fc_in_len, hidden_layer_width)
        self.fc2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.out_layer = nn.Linear(hidden_layer_width, action_length)

    def forward(self, state):

        # split statespace into respective maps based on each side of the
        # statespace maps (squares with sides = self.map_side)
        state_ = state.reshape(-1, state.shape[-1])

        in_obs = state_[:, :self.map_side ** 2]
        in_obs = in_obs.reshape(-1, 1, self.map_side, self.map_side)

        in_persons = state_[:, self.map_side ** 2:2 * self.map_side ** 2]
        in_persons = in_persons.reshape(-1, 1, self.map_side, self.map_side)

        in_goals = state_[:, 2 * self.map_side ** 2:3 * self.map_side ** 2]
        in_goals = in_goals.reshape(-1, 1, self.map_side, self.map_side)

        # convolutional processsing
        out_obs = self.pool(F.leaky_relu(self.conv1_obstacles(in_obs)))
        out_obs = self.pool(F.leaky_relu(self.conv2_obstacles(out_obs)))
        out_obs = out_obs.squeeze(dim=1)

        out_persons = self.pool(F.leaky_relu(self.conv1_persons(in_persons)))
        out_persons = self.pool(F.leaky_relu(self.conv2_persons(out_persons)))
        out_persons = out_persons.squeeze(dim=1)

        out_goals = self.pool(F.leaky_relu(self.conv1_goals(in_goals)))
        out_goals = self.pool(F.leaky_relu(self.conv2_goals(out_goals)))
        out_goals = out_goals.squeeze(dim=1)

        x = torch.cat(
            (
                out_obs.flatten(start_dim=1),
                out_persons.flatten(start_dim=1),
                out_goals.flatten(start_dim=1),
                state_[:, -4:]
            ),
            dim=1
        )

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out_layer(x)

        return x

def main():
    """Main function, run experiment."""

    tbx_writer = SummaryWriter(comment='_alpha_' + str(args.log_alpha))

    env = EwapGridworld(ped_id=1, render=args.render)

    state_size = env.reset().shape[0]
    map_side = (1 + env.vision_radius * 2)

    conv_q_net = ConvQNet(
        state_size,
        env.action_space.n,
        4096,
        map_side,
        kernel_shape=(3, 3)
    )

    soft_ac = SoftActorCritic(
        env,
        replay_buffer_size=args.replay_buffer_size,
        buffer_sample_size=args.replay_buffer_sample_size,
        tbx_writer=tbx_writer,
        tau=0.005,
        log_alpha=args.log_alpha,
        entropy_tuning=True,
        q_net=conv_q_net,
        entropy_target=args.entropy_target
    )

    soft_ac.train_and_play(args.max_episodes, args.play_interval)


if __name__ == "__main__":
    main()

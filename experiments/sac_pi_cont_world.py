import sys
from tensorboardX import SummaryWriter

sys.path.insert(0, "..")  # NOQA: E402

from rlmethods.soft_ac_pi import SoftActorCritic
from rlmethods.rlutils import ReplayBuffer
from argparse import ArgumentParser
from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speed
from envs.gridworld_drone import GridWorldDrone
from neural_nets.base_network import Checkpointer
from irlmethods.deep_maxent import RewardNet

parser = ArgumentParser()

parser.add_argument("replay_buffer_size", type=int)
parser.add_argument("replay_buffer_sample_size", type=int)
parser.add_argument("--log-alpha", type=float, default=-2.995)
parser.add_argument("--entropy-target", type=float, default=0.008)
parser.add_argument("--max-episode-length", type=int, default=10 ** 4)
parser.add_argument("--play-interval", type=int, default=1)
parser.add_argument("--rl-episodes", type=int, default=10 ** 4)
parser.add_argument("--render", action="store_true")
parser.add_argument(
    "--reward-path",
    type=str,
    default=None,
    help="The path to the reward network to be used for \
                    the training.",
)

parser.add_argument(
    "--reward-net-hidden-dims",
    type=int,
    nargs="*",
    default=[],
    help="The size of the hidden layers of the reward \
                        network.",
)

parser.add_argument(
    "--annotation-file",
    type=str,
    default=None,
    help="The location of the annotation file to \
                    be used to run the environment.",
)
parser.add_argument(
    "--checkpoint-path",
    type=str,
    default=None,
    help="Path to checkpoint to continue training from.",
)

args = parser.parse_args()


agent_width = 10
step_size = 2
obs_width = 10
grid_size = 10


def main():

    # initalize summary writer
    tbx_writer = SummaryWriter(comment="_alpha_" + str(args.log_alpha))

    # initialize replay buffer
    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    # initialize feature extractor
    feature_extractor = DroneFeatureRisk_speed(
        agent_width=agent_width,
        obs_width=obs_width,
        step_size=step_size,
        grid_size=grid_size,
        thresh1=18,
        thresh2=30,
    )

    # initialize checkpoint
    if args.checkpoint_path:
        checkpointer = Checkpointer.load_checkpointer(args.checkpoint_path)
    else:
        checkpointer = None

    # initialize environment
    env = GridWorldDrone(
        display=args.render,
        is_random=True,
        rows=576,
        cols=720,
        agent_width=agent_width,
        step_size=step_size,
        obs_width=obs_width,
        width=grid_size,
        annotation_file=args.annotation_file,
        external_control=True,
        continuous_action=True,
        consider_heading=True,
        is_onehot=False,
    )

    # initialize the reward network
    state_size = feature_extractor.extract_features(env.reset()).shape[0]
    reward_net = None
    if args.reward_path is not None:

        reward_net = RewardNet(state_size, args.reward_net_hidden_dims)
        reward_net.load(args.reward_path)

    # intialize the RL method
    soft_ac = SoftActorCritic(
        env,
        replay_buffer,
        feature_extractor,
        buffer_sample_size=args.replay_buffer_sample_size,
        tbx_writer=tbx_writer,
        tau=0.005,
        log_alpha=args.log_alpha,
        entropy_tuning=True,
        entropy_target=args.entropy_target,
        render=args.render,
        play_interval=args.play_interval,
        checkpointer=checkpointer,
    )
    soft_ac.train(
        args.rl_episodes, args.max_episode_length, reward_network=reward_net
    )

    soft_ac.policy.save("./cont_world_policies")


if __name__ == "__main__":
    main()

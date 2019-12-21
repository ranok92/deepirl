import sys
from tensorboardX import SummaryWriter

sys.path.insert(0, "..")  # NOQA: E402

from rlmethods.soft_ac_pi import SoftActorCritic
from rlmethods.rlutils import ReplayBuffer
from argparse import ArgumentParser
from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speed
from envs.gridworld_drone import GridWorldDrone

parser = ArgumentParser()
parser.add_argument("replay_buffer_size", type=int)
parser.add_argument("replay_buffer_sample_size", type=int)
parser.add_argument("--log-alpha", type=float, default=-2.995)
parser.add_argument("--entropy-target", type=float, default=0.008)
parser.add_argument("--max-episode-length", type=int, default=10 ** 4)
parser.add_argument("--play-interval", type=int, default=1)
parser.add_argument("--rl-episodes", type=int, default=10 ** 4)
parser.add_argument("--render", action="store_true")

args = parser.parse_args()


def main():

    tbx_writer = SummaryWriter(comment="_alpha_" + str(args.log_alpha))

    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    feature_extractor = DroneFeatureRisk_speed()

    env = GridWorldDrone(continuous_action=True, display=args.render)

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
    )

    soft_ac.train(args.rl_episodes, args.max_episode_length)

    soft_ac.policy.save("./cont_world_policies")


if __name__ == "__main__":
    main()

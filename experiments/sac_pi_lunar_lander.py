import sys
from tensorboardX import SummaryWriter
sys.path.insert(0, '..')  # NOQA: E402

from rlmethods.soft_ac_pi import SoftActorCritic 
from rlmethods.rlutils import ReplayBuffer
from argparse import ArgumentParser
from featureExtractor.gym_feature_extractor import IdentityFeatureExtractor
import gym

parser = ArgumentParser()
parser.add_argument('replay_buffer_size', type=int)
parser.add_argument('replay_buffer_sample_size', type=int)
parser.add_argument('--log-alpha', type=float, default=-2.995)
parser.add_argument('--entropy-target', type=float, default=0.008)
parser.add_argument('--max-episode-length', type=int, default=10**6)

args = parser.parse_args()

def main():

    tbx_writer = SummaryWriter(comment='alpha_'+str(args.log_alpha))

    env = gym.make('LunarLanderContinuous-v2')

    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    feature_extractor = IdentityFeatureExtractor()

    soft_ac = SoftActorCritic(
        env,
        replay_buffer,
        args.max_episode_length,
        feature_extractor,
        buffer_sample_size=args.replay_buffer_sample_size,
        tbx_writer = tbx_writer,
        tau=0.005,
        log_alpha=args.log_alpha,
        entropy_tuning=True,
        entropy_target=args.entropy_target
    )

    soft_ac.train_and_play(5000, 1)

if __name__ == "__main__":
    main()

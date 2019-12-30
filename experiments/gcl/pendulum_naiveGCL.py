""" Run naive GCL on a gym pendulum expert. """
import sys

sys.path.insert(0, "../..")

from argparse import ArgumentParser
import gym

from irlmethods.GCL_family import NaiveGCL, PolicyExpert
from rlmethods.soft_ac_pi import (
    SoftActorCritic,
    PolicyNetwork,
    NN_HIDDEN_WIDTH,
)
from rlmethods.rlutils import ReplayBuffer
from featureExtractor.gym_feature_extractor import IdentityFeatureExtractor
from tensorboardX import SummaryWriter

arg_parser = ArgumentParser()

arg_parser.add_argument(
    "--replay-buffer-length",
    type=int,
    required=True,
    help="Length of replay buffer for SAC.",
)

arg_parser.add_argument(
    "--replay-buffer-sample-size",
    type=int,
    required=True,
    help="number of transitions sampled from buffer.",
)

arg_parser.add_argument(
    "--irl-episodes",
    type=int,
    required=True,
    help="Number of IRL iterations.",
)

arg_parser.add_argument(
    "--disable-entropy-tuning",
    action="store_false",
    help="disable entropy tuning for SAC.",
)

arg_parser.add_argument(
    "--entropy_target",
    type=float,
    default=-2.0,
    help="Target entropy of SAC's final policy.",
)


arg_parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    help="Q-function exponential averageing, Q_new= (1-tau)*Q_old + tau*Q_new",
)

arg_parser.add_argument(
    "--play-interval",
    type=int,
    default=1,
    help="Number of SAC updates before new trajs are sampled",
)


arg_parser.add_argument(
    "--log-alpha",
    type=float,
    default=-2.995,
    help="log of initial alpha coefficient.",
)

arg_parser.add_argument(
    "--num-expert-trajs",
    type=int,
    default=50,
    help="Number of expert trajectories from policy expert.",
)

arg_parser.add_argument(
    "--max-env-steps",
    type=int,
    default=1000,
    help="Maximum allowed steps each episode in env.",
)

arg_parser.add_argument(
    "--irl-traj-per-ep",
    type=int,
    default=10,
    help="Number of policy trajectory samples per irl episode.",
)

arg_parser.add_argument(
    "--irl-num-policy-updates",
    type=int,
    default=10,
    help="Number of policy training iterations inside inner IRL loop.",
)

args = arg_parser.parse_args()


def main():
    """ Run the experiment. """

    # TensorboardX
    tbx_writer = SummaryWriter(comment="pendulum_naive_gcl_")

    tbx_writer.add_hparams(vars(args), {})

    # env related
    env = gym.make("Pendulum-v0")

    feature_extractor = IdentityFeatureExtractor()

    state_size = feature_extractor.extract_features(env.reset()).shape[0]

    # rl related
    replay_buffer = ReplayBuffer(args.replay_buffer_length)

    rl = SoftActorCritic(
        env,
        replay_buffer,
        feature_extractor,
        args.replay_buffer_sample_size,
        entropy_target=args.entropy_target,
        entropy_tuning=args.disable_entropy_tuning,
        tau=args.tau,
        log_alpha=args.log_alpha,
        play_interval=args.play_interval,
        tbx_writer=tbx_writer,
    )

    # irl related

    expert_policy = PolicyNetwork(
        state_size, env.action_space, NN_HIDDEN_WIDTH
    )
    expert_policy.load("../pendulum_policies/5.pt")
    expert = PolicyExpert(
        expert_policy, env, args.num_expert_trajs, args.max_env_steps
    )

    expert_states = expert.get_expert_states()
    expert_actions = expert.get_expert_actions()

    irl = NaiveGCL(
        rl, env, expert_states, expert_actions, tbx_writer=tbx_writer
    )

    irl.train(
        args.irl_episodes,
        args.irl_traj_per_ep,
        args.max_env_steps,
        args.irl_num_policy_updates,
    )

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()

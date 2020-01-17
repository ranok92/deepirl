""" run general deep maxent (Wulfmeier et. al) on drone environment. """
import sys
import time
import os
import argparse
import matplotlib

sys.path.insert(0, "..")  # NOQA: E402
from envs.gridworld_drone import GridWorldDrone as GridWorld
from irlmethods.irlUtils import read_expert_states
from logger.logger import Logger
import utils

from featureExtractor.drone_feature_extractor import (
    DroneFeatureSAM1,
    DroneFeatureRisk,
    DroneFeatureRisk_v2,
)
from featureExtractor.gridworld_featureExtractor import (
    FrontBackSide,
    LocalGlobal,
    OneHot,
    SocialNav,
    FrontBackSideSimple,
)
from featureExtractor.drone_feature_extractor import (
    DroneFeatureRisk_speed,
    DroneFeatureRisk_speedv2,
)


import datetime
from logger.logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--policy-path", type=str, nargs="?", default=None)
parser.add_argument(
    "--play", action="store_true", help="play given or latest stored policy."
)
parser.add_argument(
    "--dont-save",
    action="store_true",
    help="don't save the policy network weights.",
)
parser.add_argument("--render", action="store_true", help="show the env.")
parser.add_argument(
    "--on-server",
    action="store_true",
    help="True if the code is being run on a server.",
)
parser.add_argument(
    "--store-train-results",
    action="store_true",
    help="True if you want to store intermediate results",
)
parser.add_argument(
    "--store-interval",
    action="store_true",
    help="Interval of storing the results.",
)
parser.add_argument("--rl-episodes", type=int, default=50)
parser.add_argument("--rl-ep-length", type=int, default=30)
parser.add_argument("--irl-iterations", type=int, default=100)
parser.add_argument("--rl-log-intervals", type=int, default=10)

parser.add_argument(
    "--regularizer", type=float, default=0, help="The regularizer to use."
)

parser.add_argument("--seed", type=int, default=7, help="The seed for the run")

parser.add_argument(
    "--save-folder",
    type=str,
    default=None,
    required=True,
    help="The name of the directory to store the results in. The name will be used to \
                    save the plots, the policy and the reward networks.(Relative path)",
)

parser.add_argument(
    "--exp-trajectory-path",
    type=str,
    required=True,
    default=None,
    help="The name of the directory in which \
                    the expert trajectories are stored.(Relative path)",
)

parser.add_argument(
    "--feat-extractor",
    type=str,
    required=True,
    default=None,
    help="The name of the \
                     feature extractor to be used in the experiment.",
)

parser.add_argument(
    "--reward-net-hidden-dims",
    nargs="*",
    type=int,
    default=[128],
    help="The dimensions of the \
                     hidden layers of the reward network.",
)

parser.add_argument(
    "--policy-net-hidden-dims",
    nargs="*",
    type=int,
    default=[128],
    help="The dimensions of the \
                     hidden layers of the policy network.",
)

parser.add_argument(
    "--annotation-file",
    type=str,
    default=None,
    help="The location of the annotation file to \
                    be used to run the environment.",
)

parser.add_argument(
    "--lr-rl",
    type=float,
    default=1e-3,
    help="The learning rate for the policy network.",
)

parser.add_argument(
    "--lr-irl",
    type=float,
    default=1e-3,
    help="The learning rate for the reward network.",
)

parser.add_argument(
    "--clipping-value",
    type=float,
    default=None,
    help="For gradient clipping of the \
                    reward network.",
)

parser.add_argument(
    "--scale-svf",
    action="store_true",
    default=None,
    help="If true, will scale the states \
                    based on the reward the trajectory got.",
)

parser.add_argument("--replace-subject", action="store_true", default=None)
parser.add_argument(
    "--segment-size",
    type=int,
    default=None,
    help="Size of each trajectory segment.",
)
parser.add_argument("--subject", type=int, default=None)


parser.add_argument(
    "--rl-method",
    type=str,
    default="ActorCritic",
    help="The RL trainer to be used.",
)
parser.add_argument("--play-interval", type=int, default=100)
parser.add_argument("--replay-buffer-sample-size", type=int, default=1000)
parser.add_argument("--replay-buffer-size", type=int, default=5000)

parser.add_argument("--num-trajectory-samples", type=int, default=100)


def main():
    args = parser.parse_args()

    if args.on_server:
        # matplotlib without monitor
        matplotlib.use("Agg")

        # pygame without monitor
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H:%M:%S")

    policy_net_dims = "-policy_net-"
    for dim in args.policy_net_hidden_dims:
        policy_net_dims += str(dim)
        policy_net_dims += "-"

    reward_net_dims = "-reward_net-"
    for dim in args.reward_net_hidden_dims:
        reward_net_dims += str(dim)
        reward_net_dims += "-"

    parent_dir = (
        "./results/"
        + str(args.save_folder)
        + st
        + policy_net_dims
        + reward_net_dims
    )

    to_save = (
        "./results/"
        + str(args.save_folder)
        + st
        + policy_net_dims
        + reward_net_dims
        + "-reg-"
        + str(args.regularizer)
        + "-seed-"
        + str(args.seed)
        + "-lr-"
        + str(args.lr_irl)
    )

    log_file = "Experiment_info.txt"

    experiment_logger = Logger(to_save, log_file)
    experiment_logger.log_header("Arguments for the experiment :")
    experiment_logger.log_info(vars(args))

    from rlmethods.b_actor_critic import ActorCritic
    from rlmethods.soft_ac_pi import SoftActorCritic
    from rlmethods.rlutils import ReplayBuffer

    from irlmethods.general_deep_maxent import GeneralDeepMaxent, play

    agent_width = 10
    step_size = 2
    obs_width = 10
    grid_size = 10

    # check for the feature extractor being used
    # initialize feature extractor
    if args.feat_extractor == "Onehot":
        feat_ext = OneHot(grid_rows=10, grid_cols=10)
    if args.feat_extractor == "SocialNav":
        feat_ext = SocialNav()
    if args.feat_extractor == "FrontBackSideSimple":
        feat_ext = FrontBackSideSimple(
            thresh1=1,
            thresh2=2,
            thresh3=3,
            thresh4=4,
            step_size=step_size,
            agent_width=agent_width,
            obs_width=obs_width,
        )

    if args.feat_extractor == "LocalGlobal":
        feat_ext = LocalGlobal(
            window_size=5,
            grid_size=grid_size,
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
        )

    if args.feat_extractor == "DroneFeatureSAM1":

        feat_ext = DroneFeatureSAM1(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=5,
            thresh2=10,
        )

    if args.feat_extractor == "DroneFeatureRisk":

        feat_ext = DroneFeatureRisk(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=15,
            thresh2=30,
        )

    if args.feat_extractor == "DroneFeatureRisk_v2":

        feat_ext = DroneFeatureRisk_v2(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=15,
            thresh2=30,
        )

    if args.feat_extractor == "DroneFeatureRisk_speed":

        feat_ext = DroneFeatureRisk_speed(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=10,
            thresh2=15,
        )

    if args.feat_extractor == "DroneFeatureRisk_speedv2":

        feat_ext = DroneFeatureRisk_speedv2(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            thresh1=18,
            thresh2=30,
        )

    experiment_logger.log_header("Parameters of the feature extractor :")
    experiment_logger.log_info(feat_ext.__dict__)

    # initialize the environment
    if not args.dont_save and args.save_folder is None:
        print("Specify folder to save the results.")
        exit()

    env = GridWorld(
        display=args.render,
        is_random=True,
        rows=576,
        cols=720,
        agent_width=agent_width,
        step_size=step_size,
        obs_width=obs_width,
        width=grid_size,
        subject=args.subject,
        annotation_file=args.annotation_file,
        goal_state=None,
        step_wrapper=utils.step_wrapper,
        seed=args.seed,
        replace_subject=args.replace_subject,
        segment_size=args.segment_size,
        external_control=True,
        continuous_action=False,
        reset_wrapper=utils.reset_wrapper,
        consider_heading=True,
        is_onehot=False,
    )

    experiment_logger.log_header("Environment details :")
    experiment_logger.log_info(env.__dict__)

    if args.rl_method == "ActorCritic":
        rl_method = ActorCritic(
            env,
            feat_extractor=feat_ext,
            gamma=1,
            log_interval=args.rl_log_intervals,
            max_episode_length=args.rl_ep_length,
            hidden_dims=args.policy_net_hidden_dims,
            save_folder=to_save,
            lr=args.lr_rl,
            max_episodes=args.rl_episodes,
        )

    if args.rl_method == "SAC":
        if not env.continuous_action:
            print("The action space needs to be continuous for SAC to work.")
            exit()

        replay_buffer = ReplayBuffer(args.replay_buffer_size)

        rl_method = SoftActorCritic(
            env,
            replay_buffer,
            feat_ext,
            play_interval=500,
            learning_rate=args.lr_rl,
            buffer_sample_size=args.replay_buffer_sample_size,
        )

    print("RL method initialized.")
    print(rl_method.policy)
    if args.policy_path is not None:
        rl_method.policy.load(args.policy_path)

    experiment_logger.log_header("Details of the RL method :")
    experiment_logger.log_info(rl_method.__dict__)

    expert_states = read_expert_states(args.exp_trajectory_path)

    irl_method = GeneralDeepMaxent(
        rl=rl_method,
        env=env,
        expert_states=expert_states,
        learning_rate=args.lr_irl,
        save_folder=to_save,
    )

    print("IRL method intialized.")
    print(irl_method.reward_net)

    experiment_logger.log_header("Details of the IRL method :")
    experiment_logger.log_info(irl_method.__dict__)

    irl_method.train(
        args.irl_iterations,
        args.rl_episodes,
        args.rl_ep_length,
        args.num_trajectory_samples,
        args.rl_ep_length,
    )

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()

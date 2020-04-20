""" run general deep maxent (Wulfmeier et. al) on drone environment. """
import sys
import time
import datetime
import os
import argparse
import matplotlib
import gym
import glob
import pathlib
import csv
import pandas as pd

sys.path.insert(0, "..")  # NOQA: E402
from envs.gridworld_drone import GridWorldDrone as GridWorld
from irlmethods.irlUtils import read_expert_trajectories
from irlmethods.general_deep_maxent import PerTrajGCL
from logger.logger import Logger
import utils
from featureExtractor import fe_utils
from rlmethods.b_actor_critic import ActorCritic
from rlmethods.soft_ac_pi import SoftActorCritic
from rlmethods.soft_ac import QSoftActorCritic as QSAC
from rlmethods.soft_ac import SoftActorCritic as DiscreteSAC
from rlmethods.rlutils import ReplayBuffer
from metrics import metric_utils


parser = argparse.ArgumentParser()
parser.add_argument("--policy-path", type=str, nargs="?", default=None)
parser.add_argument("--render", action="store_true", help="show the env.")
parser.add_argument("--rl-episodes", type=int, default=50)
parser.add_argument("--rl-ep-length", type=int, default=30)
parser.add_argument("--irl-iterations", type=int, default=100)
parser.add_argument("--rl-log-intervals", type=int, default=10)

parser.add_argument(
    "--regularizer", type=float, default=0, help="The l2 regularizer to use."
)

parser.add_argument("--seed", type=int, default=7, help="The seed for the run")

parser.add_argument(
    "--save-folder",
    type=str,
    default=None,
    required=True,
    help="Relative path to save folder. If save folder exists, and continue \
    option is selected, training will resume.",
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
    default="../envs/expert_datasets/university_students/annotation/processed/frame_skip_1/students003_processed_corrected.txt",
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
parser.add_argument("--play-interval", type=int, default=1)
parser.add_argument("--replay-buffer-sample-size", type=int, default=1000)
parser.add_argument("--replay-buffer-size", type=int, default=5000)
parser.add_argument("--entropy-target", type=float, default=0.3)
parser.add_argument("--tau", type=float, default=0.05)
parser.add_argument("--reset-training", action="store_true")
parser.add_argument("--account-for-terminal-state", action="store_true")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument(
    "--stochastic-sampling",
    action="store_true",
    help="Whether to use stochastic policy to sample trajectories for IRL.",
)

parser.add_argument("--num-expert-samples", type=int, default=32)
parser.add_argument("--num-policy-samples", type=int, default=32)
parser.add_argument("--save-dir", type=str, default="./results")
parser.add_argument("--pre-train-iterations", type=int, default=0)
parser.add_argument("--pre-train-rl-iterations", type=int, default=0)
parser.add_argument(
    "--saving-interval",
    type=int,
    default=10,
    help="interval at which IRL saves its models.",
)
parser.add_argument("--pedestrian-width", type=float, default=10.0)


def main():
    """Runs experiment"""

    args = parser.parse_args()

    utils.seed_all(args.seed)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H:%M:%S")

    to_save = pathlib.Path(args.save_dir)
    dir_name = args.save_folder + "_" + st
    to_save = to_save / dir_name
    to_save = str(to_save.resolve())

    log_file = "Experiment_info.txt"

    experiment_logger = Logger(to_save, log_file)
    experiment_logger.log_header("Arguments for the experiment :")
    experiment_logger.log_info(vars(args))

    feat_ext = fe_utils.load_feature_extractor(args.feat_extractor)

    experiment_logger.log_header("Parameters of the feature extractor :")
    experiment_logger.log_info(feat_ext.__dict__)

    env = GridWorld(
        display=args.render,
        is_random=False,
        rows=576,
        cols=720,
        agent_width=args.pedestrian_width,
        step_size=2,
        obs_width=args.pedestrian_width,
        width=10,
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
        show_orientation=True,
        show_comparison=True,
        show_trail=True,
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

    if args.rl_method == "discrete_QSAC":
        if not isinstance(env.action_space, gym.spaces.Discrete):
            print("discrete SAC requires a discrete action space to work.")
            exit()

        replay_buffer = ReplayBuffer(args.replay_buffer_size)

        rl_method = QSAC(
            env,
            replay_buffer,
            feat_ext,
            args.replay_buffer_sample_size,
            learning_rate=args.lr_rl,
            entropy_tuning=True,
            entropy_target=args.entropy_target,
            play_interval=args.play_interval,
            tau=args.tau,
            gamma=args.gamma,
        )

    if args.rl_method == "discrete_SAC":
        if not isinstance(env.action_space, gym.spaces.Discrete):
            print("discrete SAC requires a discrete action space to work.")
            exit()

        replay_buffer = ReplayBuffer(args.replay_buffer_size)

        rl_method = DiscreteSAC(
            env,
            replay_buffer,
            feat_ext,
            args.replay_buffer_sample_size,
            learning_rate=args.lr_rl,
            entropy_tuning=True,
            entropy_target=args.entropy_target,
            play_interval=args.play_interval,
            tau=args.tau,
            gamma=args.gamma,
        )

    print("RL method initialized.")
    print(rl_method.policy)
    if args.policy_path is not None:
        rl_method.policy.load(args.policy_path)

    import pdb; pdb.set_trace()

    experiment_logger.log_header("Details of the RL method :")
    experiment_logger.log_info(rl_method.__dict__)

    expert_trajectories = read_expert_trajectories(args.exp_trajectory_path)

    irl_method = PerTrajGCL(
        rl=rl_method,
        env=env,
        expert_trajectories=expert_trajectories,
        learning_rate=args.lr_irl,
        l2_regularization=args.regularizer,
        save_folder=to_save,
        saving_interval=args.saving_interval,
    )

    print("IRL method intialized.")
    print(irl_method.reward_net)

    experiment_logger.log_header("Details of the IRL method :")
    experiment_logger.log_info(irl_method.__dict__)

    irl_method.pre_train(
        args.pre_train_iterations,
        args.num_expert_samples,
        account_for_terminal_state=args.account_for_terminal_state,
        gamma=args.gamma,
    )

    rl_method.train(
        args.pre_train_rl_iterations,
        args.rl_ep_length,
        reward_network=irl_method.reward_net,
    )

    # save intermediate RL result
    rl_method.policy.save(to_save + "/policy")

    irl_method.train(
        args.irl_iterations,
        args.rl_episodes,
        args.rl_ep_length,
        args.rl_ep_length,
        reset_training=args.reset_training,
        account_for_terminal_state=args.account_for_terminal_state,
        gamma=args.gamma,
        stochastic_sampling=args.stochastic_sampling,
        num_expert_samples=args.num_expert_samples,
        num_policy_samples=args.num_policy_samples,
    )

    metric_applicator = metric_utils.LTHMP2020()
    metric_results = metric_utils.collect_trajectories_and_metrics(
        env,
        feat_ext,
        rl_method.policy,
        len(expert_trajectories),
        args.rl_ep_length,
        metric_applicator,
    )

    pd_metrics = pd.DataFrame(metric_results).T
    pd_metrics = pd_metrics.applymap(lambda x: x[0])
    pd_metrics.to_pickle(to_save + "/metrics.pkl")

    with open(to_save + "/rl_data.csv", "a") as f:
        rl_method.data_table.write_csv(f)

    with open(to_save + "/irl_data.csv", "a") as f:
        irl_method.data_table.write_csv(f)

    with open(to_save + "/pre_irl_data.csv", "a") as f:
        irl_method.pre_data_table.write_csv(f)


if __name__ == "__main__":
    main()

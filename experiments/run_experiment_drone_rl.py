import pdb
import sys  # NOQA

sys.path.insert(0, "..")  # NOQA: E402

import numpy as np
import argparse
import torch.multiprocessing as mp
import os


import gym
import glob
from logger.logger import Logger
import matplotlib
import datetime, time

# from debugtools import compile_results
from utils import step_wrapper, reset_wrapper

parser = argparse.ArgumentParser()
parser.add_argument("--policy-path", type=str, nargs="?", default=None)
parser.add_argument("--reward-path", type=str, nargs="?", default=None)
parser.add_argument(
    "--play", action="store_true", help="play given or latest stored policy."
)
parser.add_argument(
    "--play-user", action="store_true", help="lets the user play the game"
)
parser.add_argument(
    "--dont-save",
    action="store_true",
    help="don't save the policy network weights.",
)
parser.add_argument("--render", action="store_true", help="show the env.")
parser.add_argument("--num-trajs", type=int, default=10)
parser.add_argument("--view-reward", action="store_true")

parser.add_argument(
    "--policy-net-hidden-dims", nargs="*", type=int, default=[128]
)
parser.add_argument(
    "--reward-net-hidden-dims", nargs="*", type=int, default=[128]
)

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--on-server", action="store_true")

parser.add_argument(
    "--feat-extractor",
    type=str,
    default=None,
    help="The name of the \
                     feature extractor to be used in the experiment.",
)
parser.add_argument(
    "--save-folder",
    type=str,
    default=None,
    help="The name of the folder to \
                    store experiment related information.",
)

parser.add_argument(
    "--annotation-file",
    type=str,
    default="../envs/expert_datasets/university_students/annotation/processed/frame_skip_1/students003_processed_corrected.txt",
    help="The location of the annotation file to \
                    be used to run the environment.",
)

parser.add_argument(
    "--total-episodes", type=int, default=1000, help="Total episodes of RL"
)
parser.add_argument(
    "--max-ep-length",
    type=int,
    default=200,
    help="Max length of a single episode.",
)

parser.add_argument("--replace-subject", action="store_true")
parser.add_argument("--seed", type=int, default=789)

parser.add_argument(
    "--subject",
    type=int,
    default=None,
    help="The id of the pedestrian to replace during training or \
                    testing.",
)

parser.add_argument(
    "--exp-trajectory-path",
    type=str,
    default=None,
    help="The name of the directory in which \
                    the expert trajectories are stored.(Relative path)",
)

parser.add_argument(
    "--segment-size",
    type=int,
    default=None,
    help="Size of each trajectory segment.",
)
parser.add_argument(
    "--rl-method",
    type=str,
    default="ActorCritic",
    help="The RL trainer to be used.",
)
parser.add_argument("--play-interval", type=int, default=100)
parser.add_argument("--replay-buffer-sample-size", type=int, default=1000)
parser.add_argument("--replay-buffer-size", type=int, default=5000)
parser.add_argument("--entropy-target", type=float, default=0.3)


def main():

    #####for the logger
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    ###################

    args = parser.parse_args()

    if args.on_server:

        matplotlib.use("Agg")
        # pygame without monitor
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    from matplotlib import pyplot as plt

    mp.set_start_method("spawn")

    from rlmethods.b_actor_critic import ActorCritic
    from rlmethods.soft_ac import SoftActorCritic
    from rlmethods.rlutils import ReplayBuffer

    from envs.gridworld_drone import GridWorldDrone
    from featureExtractor.drone_feature_extractor import (
        DroneFeatureSAM1,
        DroneFeatureOccup,
        DroneFeatureRisk,
        DroneFeatureRisk_v2,
        VasquezF1,
        VasquezF2,
        VasquezF3,
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

    save_folder = None

    if not args.dont_save and not args.play:

        if not args.save_folder:
            print("Provide save folder.")
            exit()

        policy_net_dims = "-policy_net-"
        for dim in args.policy_net_hidden_dims:
            policy_net_dims += str(dim)
            policy_net_dims += "-"

        reward_net_dims = "-reward_net-"
        for dim in args.reward_net_hidden_dims:
            reward_net_dims += str(dim)
            reward_net_dims += "-"

        save_folder = (
            "./results/"
            + args.save_folder
            + st
            + args.feat_extractor
            + "-seed-"
            + str(args.seed)
            + policy_net_dims
            + reward_net_dims
            + "-total-ep-"
            + str(args.total_episodes)
            + "-max-ep-len-"
            + str(args.max_ep_length)
        )

        experiment_logger = Logger(save_folder, "experiment_info.txt")
        experiment_logger.log_header("Arguments for the experiment :")
        experiment_logger.log_info(vars(args))

    window_size = 9
    step_size = 2
    agent_width = 10
    obs_width = 10
    grid_size = 10

    feat_ext = None
    # initialize the feature extractor to be used
    if args.feat_extractor == "Onehot":
        feat_ext = OneHot(grid_rows=10, grid_cols=10)
    if args.feat_extractor == "SocialNav":
        feat_ext = SocialNav(fieldList=["agent_state", "goal_state"])
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
            window_size=11,
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
            thresh1=15,
            thresh2=30,
        )

    if args.feat_extractor == "DroneFeatureOccup":

        feat_ext = DroneFeatureOccup(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            window_size=window_size,
        )

    if args.feat_extractor == "DroneFeatureRisk":

        feat_ext = DroneFeatureRisk(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            show_agent_persp=False,
            thresh1=15,
            thresh2=30,
        )

    if args.feat_extractor == "DroneFeatureRisk_v2":

        feat_ext = DroneFeatureRisk_v2(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            show_agent_persp=False,
            thresh1=15,
            thresh2=30,
        )

    if args.feat_extractor == "DroneFeatureRisk_speed":

        feat_ext = DroneFeatureRisk_speed(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            show_agent_persp=False,
            return_tensor=False,
            thresh1=10,
            thresh2=15,
        )

    if args.feat_extractor == "DroneFeatureRisk_speedv2":

        feat_ext = DroneFeatureRisk_speedv2(
            agent_width=agent_width,
            obs_width=obs_width,
            step_size=step_size,
            grid_size=grid_size,
            show_agent_persp=False,
            return_tensor=False,
            thresh1=18,
            thresh2=30,
        )

    if args.feat_extractor == 'VasquezF1':
        feat_ext = VasquezF1(agent_width*10, 0.5, 1.0)

    if args.feat_extractor == 'VasquezF2':
        feat_ext = VasquezF1(agent_width*10, 0.5, 1.0)

    if args.feat_extractor == 'VasquezF3':
        feat_ext = VasquezF3(agent_width)


    if feat_ext is None:
        print("Please enter proper feature extractor!")
        exit()
    # log feature extractor info

    if not args.dont_save and not args.play:

        experiment_logger.log_header("Parameters of the feature extractor :")
        experiment_logger.log_info(feat_ext.__dict__)

    # initialize the environment
    if args.replace_subject:
        replace_subject = True
    else:
        replace_subject = False

    env = GridWorldDrone(
        display=args.render,
        is_onehot=False,
        seed=args.seed,
        obstacles=None,
        show_trail=False,
        is_random=True,
        annotation_file=args.annotation_file,
        subject=args.subject,
        tick_speed=60,
        obs_width=10,
        step_size=step_size,
        agent_width=agent_width,
        replace_subject=replace_subject,
        segment_size=args.segment_size,
        external_control=True,
        step_reward=0.001,
        show_comparison=True,
        consider_heading=True,
        show_orientation=True,
        # rows=200, cols=200, width=grid_size)
        rows=576,
        cols=720,
        width=grid_size,
    )

    # env = gym.make('Acrobot-v1')
    # log environment info
    if not args.dont_save and not args.play:

        experiment_logger.log_header("Environment details :")
        experiment_logger.log_info(env.__dict__)

    # initialize RL

    if args.rl_method == "ActorCritic":
        model = ActorCritic(
            env,
            feat_extractor=feat_ext,
            gamma=1,
            log_interval=100,
            max_episode_length=args.max_ep_length,
            hidden_dims=args.policy_net_hidden_dims,
            save_folder=save_folder,
            lr=args.lr,
            max_episodes=args.total_episodes,
        )

    if args.rl_method == "SAC":

        replay_buffer = ReplayBuffer(args.replay_buffer_size)

        model = SoftActorCritic(
            env,
            replay_buffer,
            feat_ext,
            buffer_sample_size=args.replay_buffer_sample_size,
            entropy_tuning=True,
            play_interval=args.play_interval,
            entropy_target=args.entropy_target,
        )

    # log RL info
    if not args.dont_save and not args.play:

        experiment_logger.log_header("Details of the RL method :")
        experiment_logger.log_info(model.__dict__)

    if args.policy_path is not None:

        from debugtools import numericalSort

        policy_file_list = []
        reward_across_models = []
        # print(args.policy_path)
        if os.path.isfile(args.policy_path):
            policy_file_list.append(args.policy_path)
        if os.path.isdir(args.policy_path):
            policy_names = glob.glob(os.path.join(args.policy_path, "*.pt"))
            policy_file_list = sorted(policy_names, key=numericalSort)

        xaxis = np.arange(len(policy_file_list))

    if not args.play and not args.play_user:
        # no playing of any kind, so training

        if args.reward_path is None:

            if args.policy_path:
                model.policy.load(args.policy_path)

            if args.rl_method == "SAC":
                model.train(
                    args.total_episodes, args.max_ep_length
                )

            else:
                model.train()

        else:
            from irlmethods.deep_maxent import RewardNet

            state_size = feat_ext.extract_features(env.reset()).shape[0]
            reward_net = RewardNet(state_size, args.reward_net_hidden_dims)
            reward_net.load(args.reward_path)
            print(next(reward_net.parameters()).is_cuda)
            model.train(reward_net=reward_net)

        if not args.dont_save:
            model.policy.save(save_folder + "/policy-models/")

    if args.play:
        # env.tickSpeed = 15
        from debugtools import compile_results

        xaxis = []
        counter = 1
        plt.figure(0)
        avg_reward_list = []
        frac_good_run_list = []
        print(policy_file_list)
        for policy_file in policy_file_list:

            print("Playing for policy :", policy_file)
            model.policy.load(policy_file)
            policy_folder = policy_file.strip().split("/")[0:-2]
            save_folder = ""
            for p in policy_folder:
                save_folder = save_folder + p + "/"

            print("The final save folder ", save_folder)
            # env.tickSpeed = 10
            assert args.policy_path is not None, "pass a policy to play from!"
            if args.exp_trajectory_path is not None:
                from irlmethods.irlUtils import calculate_expert_svf

                expert_svf = calculate_expert_svf(
                    args.exp_trajectory_path,
                    max_time_steps=args.max_ep_length,
                    feature_extractor=feat_ext,
                    gamma=1,
                )
            # reward_across_models.append(model.generate_trajectory(args.num_trajs, args.render))
            if args.exp_trajectory_path is None:

                if args.dont_save:
                    rewards, state_info, sub_info = model.generate_trajectory(
                        args.num_trajs, args.render
                    )
                else:
                    rewards, state_info, sub_info = model.generate_trajectory(
                        args.num_trajs,
                        args.render,
                        path=save_folder + "/agent_generated_trajectories/",
                    )
            else:

                if args.dont_save:
                    rewards, state_info, sub_info = model.generate_trajectory(
                        args.num_trajs, args.render, expert_svf=expert_svf
                    )
                else:
                    rewards, state_info, sub_info = model.generate_trajectory(
                        args.num_trajs,
                        args.render,
                        path=save_folder + "/agent_generated_trajectories/",
                        expert_svf=expert_svf,
                    )

            avg_reward, good_run_frac = compile_results(
                rewards, state_info, sub_info
            )
            # pdb.set_trace()
            avg_reward_list.append(avg_reward)
            frac_good_run_list.append(good_run_frac)
            plt.plot(avg_reward_list, c="r")
            plt.plot(frac_good_run_list, c="g")
            plt.draw()
        plt.show()

    if args.play_user:
        env.tickSpeed = 200

        model.generate_trajectory_user(
            args.num_trajs, args.render, path="./user_generated_trajectories/"
        )

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()

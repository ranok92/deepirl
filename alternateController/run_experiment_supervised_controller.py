import pdb
import sys  # NOQA

sys.path.insert(0, "..")  # NOQA: E402

import numpy as np
import argparse
import torch.multiprocessing as mp
import os

import git
from logger.logger import Logger
import datetime, time

# from debugtools import compile_results
from utils import step_wrapper, reset_wrapper
from alternateController.supervised_policy import SupervisedPolicyController
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

parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--on-server", action="store_true")

parser.add_argument(
    "--feat-extractor",
    type=str,
    default=None,
    required=True,
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
    "--total-epochs", type=int, default=5000,
    help="Total episodes of supervised training."
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=2000,
    help="Total data tuples in a single mini batch.",
)

parser.add_argument(
    "--max-ep-length",
    type=int,
    default=500,
    help="Maximum length of an episode before automatic termination.",
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
    "--is-categorical",
    action="store_true",
    help="if true, the supervised learning method creates a classifier\
based policy."
)

parser.add_argument(
    "--continuous-control",
    action="store_true",
    help="If true, the environment uses continuous control version for the \
agent."
)   

parser.add_argument(
    "--training-data-folder",
    type=str,
    default="DroneFeatureRisk_speedv2_with_actions_lag8",
    help="Name of the folder containing the data."
)

def main():
 #####for the logger
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    ###################

    args = parser.parse_args()

    from envs.gridworld_drone import GridWorldDrone

    from featureExtractor.drone_feature_extractor import (
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
            + str(args.total_epochs)
            + "-max-ep-len-"
            + str(args.max_ep_length)
        )

        experiment_logger = Logger(save_folder, "experiment_info.txt")
        experiment_logger.log_header("Arguments for the experiment :")
        repo = git.Repo(search_parent_directories=True)
        experiment_logger.log_info({'From branch : ' : repo.active_branch.name})
        experiment_logger.log_info({'Commit number : ' : repo.head.object.hexsha})
        experiment_logger.log_info(vars(args))

    window_size = 9
    step_size = 2
    agent_width = 10
    obs_width = 10
    grid_size = 10

    feat_ext = None

    # initialize the feature extractor to be used

    if args.feat_extractor == 'DroneFeatureRisk_speedv2':

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

    if feat_ext is None:
        print("Please enter proper feature extractor!")
        sys.exit()
    
    #log feature extractor information
    if not args.dont_save and not args.play:
        experiment_logger.log_header("Parameters of the feature extractor :")
        experiment_logger.log_info(feat_ext.__dict__)


    #initialize the environment

    replace_subject = False
    if args.replace_subject:
        replace_subject = True
    else:
        replace_subject = False

    continuous_action_flag = False
    if args.continuous_control:
        continuous_action_flag =True

    env = GridWorldDrone(
        display=args.render,
        seed=args.seed,
        show_trail=False,
        is_random=False,
        annotation_file=args.annotation_file,
        subject=args.subject,
        tick_speed=60,
        obs_width=10,
        step_size=step_size,
        agent_width=agent_width,
        external_control=True,
        step_reward=0.001,
        show_comparison=True,
        replace_subject=replace_subject,
        continuous_action=continuous_action_flag,         
        # rows=200, cols=200, width=grid_size)
        rows=576,
        cols=720,
        width=grid_size,
    )

    #log information about the environment

    if not args.dont_save and not args.play:
        experiment_logger.log_header("Environment details :")
        experiment_logger.log_info(env.__dict__)
    
    #initialize the controller
    
    categorical_flag = False
    output_size = 2
    if args.is_categorical:
        categorical_flag = True
        output_size = 35
    

    controller = SupervisedPolicyController(80, output_size,
                                        categorical=categorical_flag,
                                        hidden_dims=args.policy_net_hidden_dims,
                                        policy_path=args.policy_path,
                                        mini_batch_size=2000,
                                        learning_rate=args.lr,
                                        save_folder=save_folder)
    
    if not args.dont_save and not args.play:
        experiment_logger.log_header("Environment details :")
        experiment_logger.log_info(controller.__dict__)

    base_data_path = '../envs/expert_datasets/university_students/annotation/traj_info/\
frame_skip_1/students003/'
    folder_name = args.training_data_folder
    data_folder = base_data_path+folder_name
    if not args.play:

        if categorical_flag:
            controller.train(args.total_epochs, data_folder)
        else:
            controller.train_regression(args.total_epochs, data_folder)
    
    if args.play:

        controller.play_policy(args.num_trajs, 
                                env,
                                args.max_ep_length,
                                feat_ext)




if __name__ == '__main__':

    main()

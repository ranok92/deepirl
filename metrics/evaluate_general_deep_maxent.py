import argparse
import sys  # NOQA
import pathlib
from datetime import datetime

import numpy as np

import torch

sys.path.insert(0, "..")  # NOQA: E402
sys.path.insert(0, "../..")  # NOQA: E402

import pickle


import metrics
import metric_utils

from rlmethods.soft_ac import QNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--max-ep-length",
    type=int,
    default=3500,
    help="Max length of a single episode.",
)

parser.add_argument(
    "--feat-extractor",
    type=str,
    required=True,
    help="The name of the \
                     feature extractor to be used in the experiment.",
)

parser.add_argument("--dont-replace-subject", action="store_false")

parser.add_argument(
    "--annotation-file",
    type=str,
    required=True,
    help="The location of the annotation file to \
                    be used to run the environment.",
)

parser.add_argument("--reward-path", type=str, nargs="?", default=None)

parser.add_argument("--policy-path", type=str, required=True)

parser.add_argument("--output-name", type=str, default="gmaxent_eval")


def main():

    output = {}

    # parameters for the feature extractors
    step_size = 2
    agent_width = 10
    obs_width = 10
    grid_size = 10

    args = parser.parse_args()
    output["eval parameters"] = vars(args)

    # initialize environment
    from envs.gridworld_drone import GridWorldDrone

    consider_heading = True
    np.random.seed(0)
    env = GridWorldDrone(
        display=False,
        is_onehot=False,
        seed=0,
        obstacles=None,
        show_trail=True,
        is_random=False,
        subject=None,
        annotation_file=args.annotation_file,
        tick_speed=60,
        obs_width=10,
        step_size=step_size,
        agent_width=agent_width,
        external_control=True,
        dont_replace_subject=args.dont_replace_subject,
        show_comparison=True,
        consider_heading=consider_heading,
        show_orientation=True,
        rows=576,
        cols=720,
        width=grid_size,
    )

    # initialize the feature extractor
    from featureExtractor.drone_feature_extractor import (
        DroneFeatureRisk_speedv2,
    )
    from featureExtractor.drone_feature_extractor import (
        VasquezF1,
        VasquezF2,
        VasquezF3,
    )

    if args.feat_extractor == "DroneFeatureRisk_speedv2":

        feat_ext_args = {
            "agent_width": agent_width,
            "obs_width": obs_width,
            "step_size": step_size,
            "grid_size": grid_size,
            "thresh1": 18,
            "thresh2": 30,
        }

        feat_ext = DroneFeatureRisk_speedv2(**feat_ext_args)

    if args.feat_extractor == "VasquezF1":
        feat_ext_args = {
            "density_radius": 6 * agent_width,
            "lower_speed_threshold": 18,
            "upper_speed_threshold": 30,
        }

        feat_ext = VasquezF1(
            feat_ext_args["density_radius"],
            feat_ext_args["lower_speed_threshold"],
            feat_ext_args["upper_speed_threshold"],
        )

    if args.feat_extractor == "VasquezF2":
        feat_ext_args = {
            "density_radius": 6 * agent_width,
            "lower_speed_threshold": 18,
            "upper_speed_threshold": 30,
        }

        feat_ext = VasquezF2(
            feat_ext_args["density_radius"],
            feat_ext_args["lower_speed_threshold"],
            feat_ext_args["upper_speed_threshold"],
        )

    if args.feat_extractor == "VasquezF3":
        feat_ext_args = {
            "agent_width": agent_width,
        }

        feat_ext = VasquezF3(feat_ext_args["agent_width"])

    output["feature_extractor_params"] = feat_ext_args
    output["feature_extractor"] = feat_ext

    # initialize policy
    sample_state = env.reset()
    state_size = feat_ext.extract_features(sample_state).shape[0]
    policy = QNetwork(state_size, env.action_space.n, 256)
    policy.load(args.policy_path)
    policy.to(DEVICE)

    # metric parameters
    metric_applicator = metric_utils.MetricApplicator()
    metric_applicator.add_metric(metrics.compute_trajectory_smoothness)
    metric_applicator.add_metric(metrics.compute_distance_displacement_ratio)
    metric_applicator.add_metric(metrics.proxemic_intrusions, [3])
    metric_applicator.add_metric(metrics.anisotropic_intrusions, [20])
    metric_applicator.add_metric(metrics.count_collisions, [20])
    metric_applicator.add_metric(metrics.goal_reached, [10, 10])
    metric_applicator.add_metric(metrics.trajectory_length)

    # collect trajectories and apply metrics
    num_peds = len(env.pedestrian_dict.keys())
    output["metrics"] = metric_applicator.get_metrics()
    output["metric_results"] = {}

    metric_results = metric_utils.collect_trajectories_and_metrics(
        env,
        feat_ext,
        policy,
        num_peds,
        args.max_ep_length,
        metric_applicator
    )

    output["metric_results"] = metric_results

    pathlib.Path('./results/').mkdir(exist_ok=True)

    with open(
        "./results/"
        + args.output_name
        + "_"
        + datetime.now().strftime("%Y-%m-%d-%H:%M"),
        "wb",
    ) as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    main()

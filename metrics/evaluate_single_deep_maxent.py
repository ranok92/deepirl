""" Evaluates deep maxent on a single policy .pt file"""
import argparse
import sys  # NOQA
import re
import pathlib
from datetime import datetime

import numpy as np

import torch
import os

sys.path.insert(0, "..")  # NOQA: E402
sys.path.insert(0, "../..")  # NOQA: E402

import pickle

from rlmethods.b_actor_critic import Policy

import metrics
import metric_utils

from evaluate_drift_deep_maxent import agent_drift_analysis

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
feature extractor to be used in the experiment. Set to 'Raw_state' if providing \
trajectories in the form of list of state dictionaries.",
)

parser.add_argument("--dont-replace-subject", action="store_false")

parser.add_argument(
    "--annotation-file",
    type=str,
    default="../envs/expert_datasets/university_students/\
annotation/processed/frame_skip_1/\
students003_processed_corrected.txt",
    help="The location of the annotation file to \
                    be used to run the environment.",
)

parser.add_argument("--reward-path", type=str, nargs="?", default=None)

parser.add_argument("--policy-path", type=str, required=True)

parser.add_argument("--output-name", type=str, default="deep_maxent_eval")

parser.add_argument("--disregard-collisions", action="store_true")

parser.add_argument(
    "--trajectory-folder",
    type=str,
    default=None,
    help="Folder containing trajectories.\
The trajectories have to be list of dictionaires containing the raw state.",
)

parser.add_argument(
    "--drift-timesteps",
    type=lambda s: [int(t) for t in s.split(",")],
    default=100,
)


def main(args):

    output = {}

    # parameters for the feature extractors
    step_size = 2
    agent_width = 10
    obs_width = 10
    grid_size = 10

    if args.feat_extractor != "Raw_state":
        assert os.path.isfile(args.policy_path), "Could not find policy file."
        parent_dir_name = pathlib.Path(args.policy_path).parent.name
        experiment_seed = re.search(r'seed-(\d+)', parent_dir_name).group(0)
        print("Detected seed: {}".format(experiment_seed))

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
        replace_subject=args.dont_replace_subject,
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

    from featureExtractor.drone_feature_extractor import (
        Fahad,
        GoalConditionedFahad,
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

    if args.feat_extractor == "Fahad":
        feat_ext_args = {
            "inner_ring_rad": 36,
            "outer_ring_rad": 60,
            "lower_speed_threshold": 0.5,
            "upper_speed_threshold": 1.0,
        }

        feat_ext = Fahad(36, 60, 0.5, 1.0)

    if args.feat_extractor == "GoalConditionedFahad":
        feat_ext_args = {
            "inner_ring_rad": 36,
            "outer_ring_rad": 60,
            "lower_speed_threshold": 0.5,
            "upper_speed_threshold": 1.0,
        }

        feat_ext = GoalConditionedFahad(36, 60, 0.5, 1.0)

    # no features if dealing with raw trajectories
    if args.feat_extractor == "Raw_state":
        feat_ext_args = {}
        feat_ext = None

    output["feature_extractor_params"] = feat_ext_args
    output["feature_extractor"] = feat_ext

    if args.feat_extractor != "Raw_state":
        # initialize policy
        # for getting metrics from policy files

        sample_state = env.reset()
        state_size = feat_ext.extract_features(sample_state).shape[0]
        policy = Policy(state_size, env.action_space.n, [256])
        policy.load(args.policy_path)
        policy.to(DEVICE)

        # metric parameters
        metric_applicator = metric_utils.MetricApplicator()
        metric_applicator.add_metric(metrics.compute_trajectory_smoothness)
        metric_applicator.add_metric(
            metrics.compute_distance_displacement_ratio
        )
        metric_applicator.add_metric(metrics.proxemic_intrusions, [3])
        metric_applicator.add_metric(metrics.anisotropic_intrusions, [20])
        metric_applicator.add_metric(metrics.count_collisions, [10])
        metric_applicator.add_metric(metrics.goal_reached, [10, 10])
        metric_applicator.add_metric(metrics.pedestrian_hit, [10])
        metric_applicator.add_metric(metrics.trajectory_length)
        metric_applicator.add_metric(
            metrics.distance_to_nearest_pedestrian_over_time
        )
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
            metric_applicator,
            disregard_collisions=args.disregard_collisions,
        )

        output["metric_results"] = metric_results

        # drift calculation
        drift_matrix = np.zeros(
            (len(env.pedestrian_dict.keys()), len(args.drift_timesteps))
        )
        for drift_idx, drift_timestep in enumerate(args.drift_timesteps):
            ped_drifts = agent_drift_analysis(
                policy,
                "Policy_network",
                env,
                list(
                    [
                        int(ped_key)
                        for ped_key in env.pedestrian_dict.keys()
                    ]
                ),
                feat_extractor=feat_ext,
                pos_reset=drift_timestep,
            )

            assert len(ped_drifts) == len((env.pedestrian_dict.keys()))

            drift_matrix[:, drift_idx] = ped_drifts

            output["metric_results"]["drifts"] = drift_matrix
            output["metric_results"]["drifts_header"] = args.drift_timesteps

            pathlib.Path("./results/").mkdir(exist_ok=True)

        with open(
            "./results/"
            + args.output_name
            + "_"
            + parent_dir_name
            + "_"
            + pathlib.Path(args.policy_path).name
            + datetime.now().strftime("%Y-%m-%d-%H:%M"),
            "wb",
        ) as f:
            pickle.dump(output, f)
    else:
        # when raw trajectories are directly provided.
        # metric parameters
        metric_applicator = metric_utils.MetricApplicator()
        metric_applicator.add_metric(
            metrics.compute_trajectory_smoothness, [10]
        )
        metric_applicator.add_metric(
            metrics.compute_distance_displacement_ratio, [10]
        )
        metric_applicator.add_metric(metrics.proxemic_intrusions, [3])
        metric_applicator.add_metric(metrics.anisotropic_intrusions, [20])
        metric_applicator.add_metric(metrics.count_collisions, [10])
        metric_applicator.add_metric(metrics.goal_reached, [10, 10])
        metric_applicator.add_metric(metrics.pedestrian_hit, [10])
        metric_applicator.add_metric(metrics.trajectory_length)
        metric_applicator.add_metric(
            metrics.distance_to_nearest_pedestrian_over_time
        )

        metric_results = metric_utils.collect_metrics_from_trajectory(
            args.trajectory_folder, metric_applicator
        )

        output["metric_results"] = metric_results

        pathlib.Path("./results/").mkdir(exist_ok=True)

        output_filename = args.trajectory_folder.strip().split("/")[-1]
        with open(
            "./results/"
            + args.output_name
            + "_"
            + output_filename
            + "_"
            + datetime.now().strftime("%Y-%m-%d-%H:%M"),
            "wb",
        ) as f:
            pickle.dump(output, f)


if __name__ == "__main__":
    in_args = parser.parse_args()

    main(in_args)

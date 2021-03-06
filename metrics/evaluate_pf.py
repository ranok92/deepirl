
import argparse
import sys  # NOQA
import pathlib
from datetime import datetime

import numpy as np

import torch

sys.path.insert(0, "..")  # NOQA: E402
sys.path.insert(0, "../..")  # NOQA: E402

import pickle

from rlmethods.b_actor_critic import Policy
from alternateController.potential_field_controller import PotentialFieldController as PFController

import metrics
import metric_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--max-ep-length",
    type=int,
    default=3500,
    help="Max length of a single episode.",
)

parser.add_argument("--dont-replace-subject", action="store_false")

parser.add_argument(
    "--annotation-file",
    type=str,
    default='../envs/expert_datasets/university_students/\
annotation/processed/frame_skip_1/\
students003_processed_corrected.txt',
    help="The location of the annotation file to \
                    be used to run the environment.",
)

parser.add_argument("--reward-path", type=str, nargs="?", default=None)

parser.add_argument("--output-name", type=str, default="deep_maxent_eval")

parser.add_argument("--disregard-collisions", action="store_true")


def main(args):

    output = {}

    # parameters for the feature extractors
    step_size = 2
    agent_width = 10
    obs_width = 10
    grid_size = 10

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


    feat_ext_args = {}
    feat_ext = None
    '''
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

    if args.feat_extractor == 'Fahad':
        feat_ext_args = {
            "inner_ring_rad" : 36,
            "outer_ring_rad" : 60,
            "lower_speed_threshold" : 0.5,
            "upper_speed_threshold" : 1.0
        }

        feat_ext = Fahad(36, 60, 0.5, 1.0)

    if args.feat_extractor == 'GoalConditionedFahad':
        feat_ext_args = {
            "inner_ring_rad" : 36,
            "outer_ring_rad" : 60,
            "lower_speed_threshold" : 0.5,
            "upper_speed_threshold" : 1.0
        }

        feat_ext = GoalConditionedFahad(36, 60, 0.5, 1.0)



    output["feature_extractor_params"] = feat_ext_args
    output["feature_extractor"] = feat_ext
    '''
    # initialize policy
    '''
    sample_state = env.reset()
    state_size = feat_ext.extract_features(sample_state).shape[0]
    policy = Policy(state_size, env.action_space.n, [256])
    policy.load(args.policy_path)
    policy.to(DEVICE)
    '''
    orient_quant = env.orient_quantization
    orient_div = len(env.orientation_array)
    speed_div = len(env.speed_array)

    policy = PFController(speed_div, orient_div, orient_quant)

    # metric parameters
    metric_applicator = metric_utils.MetricApplicator()
    metric_applicator.add_metric(metrics.compute_trajectory_smoothness, [10])
    metric_applicator.add_metric(metrics.compute_distance_displacement_ratio, [10])
    metric_applicator.add_metric(metrics.proxemic_intrusions, [3])
    metric_applicator.add_metric(metrics.anisotropic_intrusions, [20])
    metric_applicator.add_metric(metrics.count_collisions, [10])
    metric_applicator.add_metric(metrics.goal_reached, [10, 10])
    metric_applicator.add_metric(metrics.pedestrian_hit, [10])
    metric_applicator.add_metric(metrics.trajectory_length)
    metric_applicator.add_metric(metrics.distance_to_nearest_pedestrian_over_time)
    # collect trajectories and apply metrics
    num_peds = len(env.pedestrian_dict.keys())
    output["metrics"] = metric_applicator.get_metrics()
    output["metric_results"] = {}

    metric_results = metric_utils.collect_trajectories_and_metrics_non_NN(
        env,
        policy,
        num_peds,
        args.max_ep_length,
        metric_applicator,
        disregard_collisions=args.disregard_collisions,
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
    args = parser.parse_args()

    main(args)

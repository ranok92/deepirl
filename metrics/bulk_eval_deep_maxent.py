""" Script for evaluating deep maxent in bulk """

from argparse import Namespace, ArgumentParser
from glob import glob
import re
import pathlib
from evaluate_deep_maxent import main


parser = ArgumentParser()

parser.add_argument("--feature-extractor", type=str, required=True)
parser.add_argument(
    "--parent-folder",
    type=str,
    required=True,
    help="Path to FOLDER containing all the seed folders.",
)

parser.add_argument("--identifier-name", type=str, required=True)

bulk_args = parser.parse_args()

annotation_file = "../envs/expert_datasets/university_students/annotation/processed/frame_skip_1/students003_processed_corrected.txt"

parent_path = pathlib.Path(bulk_args.parent_folder)
max_ep_length = 3500

for seed in parent_path.glob("./*"):
    seed_number = str(seed).split("/")[-1]

    policies = seed.glob("./*.pt")

    for policy in policies:

        policy_name = str(policy).split("/")[-1]
        output_name = "_".join(
            [
                bulk_args.identifier_name,
                bulk_args.feature_extractor,
                seed_number,
                policy_name,
            ]
        )

        args = Namespace(
            max_ep_length=max_ep_length,
            feat_extractor=bulk_args.feature_extractor,
            annotation_file=annotation_file,
            reward_path=None,
            policy_path=str(policy.resolve()),
            output_name=output_name,
            dont_replace_subject=False,
        )

        main(args)

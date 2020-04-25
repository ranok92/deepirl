import argparse
from tqdm import tqdm
import pathlib
import pandas as pd
import sys

sys.path.insert(0, '..')

from envs.gridworld_drone import UCYWorld
from metrics import metric_utils
from rlmethods.soft_ac import QSoftActorCritic as QSAC
from rlmethods.rlutils import ReplayBuffer
from featureExtractor.fe_utils import load_feature_extractor

parser = argparse.ArgumentParser()

parser.add_argument("--root-folder", type=str)


def main(args):
    root = pathlib.Path(args.root_folder)
    run_folders = list(root.glob("*"))

    for run_folder in tqdm(run_folders):
        policy_dir = run_folder / "policy"
        saved_policies = policy_dir.glob("*")

        if not policy_dir.exists():
            continue

        latest_policy = max(
            saved_policies,
            key=lambda x: int(str(x).split("/")[-1].split(".")[0]),
        )

        env = UCYWorld()
        fe = load_feature_extractor("GoalConditionedFahad")

        rl_method = QSAC(
            env, ReplayBuffer(1000), fe
        )

        rl_method.policy.load(str(latest_policy.resolve()))

        metric_applicator = metric_utils.LTHMP2020()

        results = metric_utils.collect_trajectories_and_metrics(
            env, fe, rl_method.policy, 430, 1000, metric_applicator
        )

        pd_metrics = pd.DataFrame(results).T
        pd_metrics = pd_metrics.applymap(lambda x: x[0])
        pd_metrics.to_pickle(str((run_folder / "metrics.pkl").resolve()))


if __name__ == "__main__":
    passed_args = parser.parse_args()
    main(passed_args)

from matplotlib import pyplot as plt
import pickle
import pathlib

import sys

sys.path.insert(0, "..")


def extract_metric(results, metric_name):
    metrics = []
    for result in results:
        result_metrics = []
        for key, val in result["metric_results"].items():
            result_metrics.append(val[metric_name])
        metrics.append(result_metrics)

    return metrics


def draw_plots():
    with open("./results/long_gdm_2020-02-15-03:20", "rb") as f:
        gdm = pickle.load(f)
    with open("./results/deep_maxent_eval_2020-02-14-16:32", "rb") as f:
        dm = pickle.load(f)

    goals_reached = extract_metric([dm, gdm], "goal_reached")

    dm_goals = sum([res for l in goals_reached[0] for res in l])
    gdm_goals = sum([res for l in goals_reached[1] for res in l])

    plt.bar([dm_goals, gdm_goals], 10)
    plt.show()


if __name__ == "__main__":
    draw_plots()


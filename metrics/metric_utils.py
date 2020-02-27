""" Utilities for generating results from metrics """

from collections import defaultdict
import copy
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MetricApplicator:
    """
    Helper class designed to apply several metrics to trajectories from any
    policies.
    """

    def __init__(self):
        self.metrics_dict = {}

    def add_metric(
        self, metric_function, metric_args=None, metric_kwargs=None
    ):
        """
        Add a metric to apply to the trajectory. The apply() function will
        call the metric function specified in the arguments of this function
        as follows:

        metric_function(*[trajectory]+metric_args, **metric_kwargs)

        :param metric_function: metric function to apply.
        :type metric_function: function.

        :param metric_args: Arguments other than trajectory to pass to the
        function. Note that these must be in order. For example, for the
        function signature metric_func(trajectory, param_a, param_b), one
        should pass [param_a, param_b] as metric_args. defaults to None
        :type metric_args: List, optional

        :param metric_kwargs: Keyword arguments for metric_function(). This
        should be a dictionary with exact mapping between keywords and
        arguments. For example, a function with signature
        metric_func(trajectory, param_a=None, param_b=None) has keyword
        dictionary {"param_a":*some value*,"param_b":*some value*}. defaults
        to None
        :type metric_kwargs: Dictionary, optional
        """
        # handle no arguments or kwargs
        metric_args = metric_args if metric_args else []
        metric_kwargs = metric_kwargs if metric_kwargs else {}

        self.metrics_dict[metric_function.__name__] = {
            "func": metric_function,
            "args": metric_args,
            "kwargs": metric_kwargs,
        }

    def apply(self, trajectories):
        """
        Apply metrics defined by add_metric to the list of trajectories
        passed in as argument.

        :param trajectories: List of trajectories to apply metrics to. For
        single trajectories, pass in [trajectory].
        :type trajectories: List

        :return: Result dictionary mapping metric to results of metric on
        trajectories.
        :rtype: dictionary.
        """

        metric_results = defaultdict(list)

        for traj in trajectories:

            for metric_name, metric_specs in self.metrics_dict.items():
                args = [traj] + metric_specs["args"]
                kwargs = metric_specs["kwargs"]
                metric_function = metric_specs["func"]

                result = metric_function(*args, **kwargs)

                metric_results[metric_name].append(result)

        return metric_results

    def get_metrics(self):
        """
        Returns metrics already added to this applicator.

        :return: dictionary of all metrics added to this applicator.
        :rtype: dictionary.
        """
        return self.metrics_dict


def collect_trajectories_and_metrics(
    env,
    feature_extractor,
    policy,
    num_trajectories,
    max_episode_length,
    metric_applicator,
):
    """
    Helper function that collects trajectories and applies metrics from a
    metric applicator on a per trajectory basis.

    :param env: environment to collect trajectories from.
    :type env: any gym-like environment.

    :param feature_extractor: a feature extractor to translate state
    dictionary to a feature vector.
    :type feature_extractor: feature extractor class.

    :param policy: Policy to extract actions from.
    :type policy: standard policy child of BasePolicy.

    :param num_trajectories: Number of trajectories to sample.
    :type num_trajectories: int.

    :param max_episode_length: Maximum length of individual trajectories.
    :type max_episode_length: int.

    :param metric_applicator: a metric applicator class containing all
    metrics that need to be applied.
    :type metric_applicator: Instance, child, or similar to
    metric_utils.MetricApplicator.

    :return: dictionary mapping trajectory to metric results from that
    trajectory.
    :rtype: dictionary
    """

    metric_results = {}

    for traj_idx in range(num_trajectories):

        print("Collecting trajectory {}".format(traj_idx))

        state = env.reset()
        done = False
        t = 0
        traj = [copy.deepcopy(state)]

        while not done and t < max_episode_length:

            print("playing timestep {}".format(t), end="\r")

            feat = feature_extractor.extract_features(state)
            feat = torch.from_numpy(feat).type(torch.FloatTensor).to(DEVICE)

            action = policy.eval_action(feat)
            state, _, done, _ = env.step(action)
            traj.append(copy.deepcopy(state))

            t += 1

        # metrics
        traj_metric_result = metric_applicator.apply([traj])
        metric_results[traj_idx] = traj_metric_result

    return metric_results
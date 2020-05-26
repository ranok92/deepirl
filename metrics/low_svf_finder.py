"""Helper script to help find lowest SVF run in all runs."""

from argparse import ArgumentParser
from tensorboard.backend.event_processing import event_accumulator

parser = ArgumentParser()

parser.add_argument(
    "--tf-file",
    type=str,
    required=True,
    help="path to the tf .events file to read and find lowest SVF from.",
)


def main(args):
    """
    Find the lowest SVF run and print it.

    :param args: arguments as parsed by argument parser.
    :type args: Argparse.namespace
    """
    ea = event_accumulator.EventAccumulator(args.tf_file)
    ea.Reload()

    svfs = ea.Scalars("Log_info/svf_difference")
    lowest_svf = sorted(svfs, key=lambda x: x[2])

    print(
        "Lowest svf is {} at step {}".format(
            lowest_svf[0].value, lowest_svf[0].step
        )
    )


if __name__ == "__main__":
    in_args = parser.parse_args()
    main(in_args)

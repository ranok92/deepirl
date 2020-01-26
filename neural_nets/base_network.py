"""defines the base network used in the project"""

import pathlib
import os
import time
import glob
from collections import namedtuple

import torch
import torch.nn as nn


def timestamp():
    """returns a timestamp of current time.

    :return: formatted timestamp string.
    :rtype: string
    """
    current_time = time.localtime()
    t = time.strftime("%Y-%m-%d_%H:%M:%S", current_time)

    return t


class BaseNN(nn.Module):
    """Base neural network, implements convenient saving and loading. All NNs
    should subclass"""

    def __init__(self):
        super(BaseNN, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def save(self, path):
        """Save the model.

        :param path: path in which to save the model.
        """
        model_i = 0

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        while os.path.exists(os.path.join(path, "%s.pt" % model_i)):
            model_i += 1

        filename = os.path.join(path, "%s.pt" % model_i)

        torch.save(self.state_dict(), filename)

    def load(self, path):
        """load the model.

        :param path: path from which to load the model.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        self.to(self.device)

    def forward(self, *inputs):
        raise NotImplementedError


class RectangleNN(BaseNN):
    """
    Neural network with rectangular hidden layers (i.e. same widths).
    """

    def __init__(self, num_layers, layer_width, activation_func):
        super(RectangleNN, self).__init__()

        self.activation_func = activation_func

        self.hidden = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden.append(nn.Linear(layer_width, layer_width))

    def hidden_forward(self, hidden_x):
        """Passes input through hidden layers.

        :param hidden_x: input.
        """
        for layer in self.hidden:
            hidden_x = self.activation_func(layer(hidden_x))

        return hidden_x

    def forward(self, *inputs):
        raise NotImplementedError


class BasePolicy(BaseNN):
    """Implements a base policy pi(a|s). All policies in RL methods should
    subclass."""

    def forward(self, *inputs):
        raise NotImplementedError

    def sample_action(self, state):
        """ Sample a random action from the policy distirbution.

        :param state: Pytorch tensor, NxS where N is batch size (could be 1)
        and S is state size.
        :type state: Pytorch tensor.
        :return action: Pytorch NxA vector where N is batch size and A is
        action dimension.
        """
        raise NotImplementedError

    def eval_action(self, state):
        """ Takes an "evaluation" action sample from policy. This evaulation
        sample should in some sense be the optimal action.

        :param state: Pytorch tensor, NxS where N is batch size (could be 1)
        and S is state size.
        :type state: Pytorch tensor.
        :return action: Pytorch NxA vector where N is batch size and A is
        action dimension.
        """

        return self.sample_action(state)


ModelTuple = namedtuple("ModelTuple", "model optimizer")


class Checkpointer:
    """Checkpoints a model and its corresponding optimizer to a timestamped
    file."""

    def __init__(
        self, folder_path, checkpoint_interval, name, create_folder=True
    ):
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_counter = 0
        self.name = name

        self.models = {}

        # figure out folder name for this run
        parent_path = pathlib.Path(folder_path)

        if create_folder:
            folder_name = "_".join(["run", self.name, timestamp()])
            self.path = parent_path / folder_name
            self.path.mkdir(
                parents=True, exist_ok=True
            )  # pylint: disable=no-member

        else:
            self.path = parent_path

    def increment_counter(self):
        """Increment counter and save if interval has elapsed."""
        self.checkpoint_counter += 1

        if self.checkpoint_counter % self.checkpoint_interval == 0:
            self.save()

    def add_model(self, model_name, model, optimizer):
        """Add model and optimizer pair to be kept track of.

        :param model_name: name of the mode, e.g. policy, value, etc.
        :type model_name: String
        :param model: model to save.
        :type model: BaseNN
        :param optimizer: Optimizer corresponding to model.
        :type optimizer: Torch optimizer.
        """

        self.models[model_name] = ModelTuple(model, optimizer)

    @staticmethod
    def latest_checkpoint_in_folder(path):
        """Find latest checkpoint in current run folder.

        :param path: Path to find latest checkpoint in
        :type path: String or pathlib.Path
        :return: Latest checkpoint .pt file.
        :rtype: file containing OrderedDict.
        """
        list_of_checkpoints = glob.glob(str(pathlib.Path(path) / "*.pt"))
        latest_checkpoint = max(list_of_checkpoints, key=os.path.getctime)

        return latest_checkpoint

    def find_latest_checkpoint(self):
        """Find latest checkpoint in current run folder.

        :return: Latest checkpoint .pt file.
        :rtype: file containing OrderedDict.
        """
        return Checkpointer.latest_checkpoint_in_folder(self.path)

    def save(self):
        """
        Save the state of model and optimizer to a file.

        :param optional_args: Optional arguments to save in file., defaults
        to None.
        :type optional_args: dictionary, optional
        """

        state = {
            "models": self.models,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_counter": self.checkpoint_counter,
            "name": self.name,
        }

        filename = (
            self.name
            + "_{}_".format(self.checkpoint_counter)
            + timestamp()
            + ".pt"
        )

        torch.save(state, self.path / filename)

    @staticmethod
    def load_checkpointer(path):
        """Loads a checkpointer using checkpoint provided in path.

        :param path: Path to checkpoint.
        :type path: String of pathlib.Path
        :return: Checkpointer object.
        """
        folder_path = pathlib.Path(path).parent
        state = torch.load(path)

        checkpointer = Checkpointer(
            folder_path,
            state["checkpoint_interval"],
            state["name"],
            create_folder=False,
        )

        checkpointer.models = state["models"]

        checkpointer.checkpoint_counter = state["checkpoint_counter"]

        return checkpointer

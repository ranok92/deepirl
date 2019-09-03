'''defines the base network used in the project'''

import pathlib
import os

import torch
import torch.nn as nn


class BaseNN(nn.Module):
    """Base neural network, implements convenient saving and loading. All NNs
    should subclass"""

    def __init__(self):
        super(BaseNN, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def save(self, path):
        """Save the model.

        :param path: path in which to save the model.
        """
        model_i = 0

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        while os.path.exists(os.path.join(path, '%s.pt' % model_i)):
            model_i += 1

        filename = os.path.join(path, '%s.pt' % model_i)

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

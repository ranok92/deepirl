'''defines the base network used in the project'''

import pathlib
import os

import torch
import torch.nn as nn


class BaseNN(nn.Module):

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

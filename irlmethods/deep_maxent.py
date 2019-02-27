
'''
Deep maxent as defined by Wulfmeier et. al.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(0, '..')
import utils  # NOQA: E402
from irlmethods.irlUtils import getStateVisitationFreq  # NOQA: E402


class RewardNet(nn.Module):
    """Policy network"""

    def __init__(self, state_dims, action_dims):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(state_dims, 128)

        self.reward_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))

        return self.reward_head(x)

    def save(self, path):
        """Save the model.

        :param path: path in which to save the model.
        """
        model_i = 0

        # os.makedirs(path, parents=True, exist_ok=True)

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        while os.path.exists(os.path.join(path, '%s.pt' % model_i)):
            model_i += 1

        filename = os.path.join(path, '%s.pt' % model_i)

        torch.save(self.state_dict(), filename)

    def load(self, path):
        """load the model.

        :param path: path from which to load the model.
        """
        self.load_state_dict(torch.load(path))
        self.eval()


class DeepMaxent():
    def __init__(rlmethod):
        self.rl = rlmethod

    def expert_svf():
        pass

    def policy_svf():
        pass

    def calculate_grads():
        pass

    def train():
        pass

'''
This file has the two networks
1. Cost function
2. Policy
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# cost network


class CostNetwork(nn.Module):
    """
    Cost (reward) network for RL function approx. methods.
    """

    def __init__(self, _nn_params):
        """__init__

        :param nn_params: a dictionary containing 3 keys : 'input' - size of
        input layer , 'hidden' - a list containing size of hidden layers
        'output' - size of output layer *for policy the size of the action
        snn_paramspace.
        """
        super(CostNetwork, self).__init__()

        self.input = _nn_params['input']
        self.output = _nn_params['output']
        self.hidden = _nn_params['hidden']
        self.no_of_hidden_layers = len(self.hidden)

        self.inputLayer = nn.Linear(self.input, self.hidden[0])
        self.hidden1 = nn.Linear(self.hidden[0], self.hidden[1])
        self.outputLayer = nn.Linear(
            self.hidden[self.no_of_hidden_layers-1], 1)

    def forward(self, x):

        #x - tensor
        x = F.elu(self.inputLayer(x))

        x = F.elu(self.hidden1(x))

        x = self.outputLayer(x)

        x = F.sigmoid(x)

        return x


class Policy(nn.Module):
    """
    Policy network for policy gradient RL methods.
    """

    def __init__(self, _nn_params):

        super(Policy, self).__init__()
        self.input = _nn_params['input']
        self.output = _nn_params['output']
        self.hidden = _nn_params['hidden']
        self.no_of_hidden_layers = len(self.hidden)

        self.inputLayer = nn.Linear(self.input, self.hidden[0])
        self.hidden1 = nn.Linear(self.hidden[0], self.hidden[1])

        self.action_head = nn.Linear(self.hidden[1], self.output)
        self.value_head = nn.Linear(self.hidden[1], 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):

        x = F.relu(self.inputLayer(x))
        #x = F.relu(self.affine2(x))
        x = F.relu(self.hidden1(x))

        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return F.softmax(action_scores, dim=-1), state_values


if __name__ == '__main__':

    nn_params = {'input': 3, 'hidden': [10], 'output': 1}
    cNN = CostNetwork(nn_params)
    inp = torch.rand(2, 3)
    y = cNN(inp)
    for i in range(2):
        print 'For iteration ', i
        y[i].backward(retain_graph=True)
        for p in cNN.parameters():

            print 'The p', p
            print 'The corresponding grad ', p.grad

""" general utilities that might find use in many seperate cases """

import numpy as np
import pandas as pd
import random
import torch
import sys
import os
import collections


class HistoryBuffer:
    """A buffer that keeps track of state visitation history."""

    def __init__(self, bufferSize=10):
        self.bufferSize = bufferSize
        self.buffer = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def addState(self, state):
        """ add a state to the history buffer each state is assumed to be of
        shape ( 1 x S ) """

        if len(self.buffer) >= self.bufferSize:
            del self.buffer[0]  # remove the oldest state

        self.buffer.append(state.cpu().numpy())

    def getHistory(self):
        """
         returns the 10 states in the buffer in the form of a torch tensor in
         the order in which they were encountered
        """

        arrSize = self.buffer[0].shape[1]
        arrayHist = np.asarray(self.buffer)

        arrayHist = np.reshape(arrayHist, (1, arrSize * self.bufferSize))
        state = torch.from_numpy(arrayHist).to(self.device)
        state = state.type(torch.cuda.FloatTensor)

        return state


def to_oh(idx, size):
    """
    creates a one-hot array of length 'size' and sets indexes in list 'idx' to
    be ones.

    params:
        idx: list or numpy array of indices to be one.
        size: size of output vector.

    return:
        output numpy vector of size ('size' x 1)
    """
    out = np.zeros(size)
    out[idx] = 1

    return out


def reset_torch_state(dtype=torch.float32):
    """decorator to return a torch tensor from gym's env.reset() function.

    :param f: function being decorated, in this case env.reset().
    """

    def real_decorator(f):
        """real decorator, taking into account the dtype.

        :param f: env.reset() being decorated.
        """

        def inner(*args, **kwargs):
            """returns a torch tensor for cpu or gpu when appropriate."""

            s = f(*args, **kwargs)

            if torch.cuda.is_available():
                return torch.from_numpy(s).cuda().type(dtype)

            return torch.from_numpy(s).type(dtype)

        return inner

    return real_decorator


def step_torch_state(dtype=torch.float32):
    """decorator to return a torch tensor from gym's env.step() function.

    :param f: function being decorated, in this case env.step().
    """

    def real_decorator(f):
        """real decorator, taking into account the dtype.

        :param f: env.step() being decorated.
        """

        def inner(*args, **kwargs):
            """returns a torch tensor for cpu or gpu when appropriate."""

            s, r, d, p = f(*args, **kwargs)

            if torch.cuda.is_available():
                s = torch.from_numpy(s).cuda().type(dtype)

            else:
                s = torch.from_numpy(s).type(dtype)

            return s, r, d, p

        return inner

    return real_decorator


def identity_dec(f):
    def same_func(*args, **kwargs):
        return f(*args, **kwargs)

    return same_func


def identity_wrapper(*output):
    if len(output) == 1:

        return output[0]
    elif len(output) == 0:
        return None

    return output


def step_wrapper(s, r, d, p, dtype=torch.float):
    if torch.cuda.is_available():
        s = torch.from_numpy(s).cuda().type(dtype)

    else:
        s = torch.from_numpy(s).type(dtype)

    return s, r, d, p


def reset_wrapper(s, dtype=torch.float):

    if torch.cuda.is_available():
        return torch.from_numpy(s).cuda().type(dtype)

    return torch.from_numpy(s).type(dtype)


def copy_dict(in_dict):
    """
    Makes a faster deep copy of a dictionary, provided it only includes
    native types, numpy arrays, and other dicts containing the
    aforementioned.

    :param in_dict: dictionary to copy.
    :type in_dict: dict.
    :return: (deep) copy of input dictionary.
    :rtype: dict.
    """

    if in_dict is None:
        return None

    out_dict = {}

    for key, val in in_dict.items():
        if isinstance(val, np.ndarray):
            out_dict[key] = val.copy()
        elif isinstance(val, dict):
            out_dict[key] = copy_dict(val)

        else:
            out_dict[key] = val

    return out_dict

def seed_all(seed):
    """ Use a seed to seed numpy, pytorch, and python random modules. """

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class DataTable:
    def __init__(self):
        self.data = collections.defaultdict(list)

    def add_row(self, data_dict, step):
        for key, val in data_dict.items():
            self.data[key].append(val)

        self.data['step'].append(step)

    def write_csv(self, file):
        pd_data = pd.DataFrame(self.data)
        pd_data.to_csv(file)

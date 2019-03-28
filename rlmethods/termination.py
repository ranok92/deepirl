import random
'''
file should contain methods and classes needed specifically to help run the rl methods
'''
import math
import collections
import numpy as np
import pdb
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '..')


class BaseTermination():

    def add_loss(self, loss):
        raise NotImplementedError

    def is_terminated(self):
        raise NotImplementedError


class VarianceTermination():

    def __init__(self, window_size=100, stop_threshold=0.2, log_interval=100):
        """Initialize a VarianceTermination instance.

        :param window_size: size of window for which samples are kept.
        :param stop_threshold: Variance threshold causing RL to stop. This is
        percent based value, between 0.0 and 1.0.
        """
        self.losses = collections.deque(maxlen=window_size)
        self.losses_sum = 0

        # create the window over which the variances will be computed.
        assert window_size > 2, 'Window must be atleast 2 elements.'
        self.window_size = window_size

        self.stop_threshold = stop_threshold

        # debug data stores
        # self.means = collections.deque(maxlen=window_size)
        # self.variances = collections.deque(maxlen=window_size)
        self.variances = []
        self.means = []
        self.log_counter = 0
        self.log_interval = log_interval

        # plt.ion()

    def is_full(self):
        """returns true if buffer is full."""
        if len(self.losses) == self.window_size:
            return True

        return False

    def add_loss(self, loss):
        """Add a loss to the buffer and perform associated processing.

        :param loss: loss to add to buffer.
        """

        if self.is_full():
            first_loss = self.losses.popleft()
            self.losses_sum -= first_loss

        self.losses.append(loss)
        self.losses_sum += loss

    def is_terminated(self):

        if not self.is_full():
            return False

        mean = self.losses_sum / len(self.losses)

        # calculate Bessel-corrected variance
        variance = sum([(loss - mean)**2 for loss in self.losses])
        variance /= len(self.losses) - 1

        if math.sqrt(variance) < self.stop_threshold * mean:
            return True

        # store debug info
        self.means.append(mean)
        self.variances.append(variance)

        return False

    def print_debug(self):
        plt.cla()

        plt.plot(self.means, 'b')
        plt.plot(self.variances, 'r')

        plt.draw()
        plt.pause(0.0001)

    def plot_avg_loss(self):
        if self.log_counter % self.log_interval  == 0:
            self.print_debug()

        self.log_counter += 1


class LossBasedTermination(BaseTermination):

    def __init__(
            self,
            list_size=100,
            stop_threshold=.5,
            info=True,
            log_interval=50
    ):

        self.loss_diff_list = collections.deque(maxlen=list_size)
        self.list_size = list_size
        self.stop_threshold = stop_threshold
        self.last_loss = None
        self.current_avg_loss = None
        self.info = info

        self.log_interval = log_interval
        if self.info:
            self.current_avg_loss_diff_list = []

    def add_loss(self, new_loss):

        if self.last_loss is None:
            self.last_loss = new_loss

        else:
            new_diff = abs(self.last_loss - new_loss)
            self.loss_diff_list.append(new_diff)
            self.last_loss = new_loss
            self.current_avg_loss = sum(self.loss_diff_list)/self.list_size

        if self.current_avg_loss is not None:
            self.current_avg_loss_diff_list.append(self.current_avg_loss)

    def is_terminated(self):
        """Returns true if RL has converged."""

        if len(self.loss_diff_list) == self.list_size:
            if self.current_avg_loss < self.stop_threshold:
                return True

        return False

    def plot_avg_loss(self):

        if len(self.current_avg_loss_diff_list) > 0 and len(self.current_avg_loss_diff_list) % self.log_interval == 0:
            plt.plot(self.current_avg_loss_diff_list)
            plt.draw()
            plt.pause(.0001)


if __name__ == '__main__':

    l = LossBasedTermination(list_size=10, stop_threshold=3, log_interval=100)
    losslist = [random.random() for _ in range(10000)]
    for val in losslist:
        l.update_list(val)
        print('avg loss :', l.current_avg_loss)
        print(l.loss_diff_list)
        print(l.check_termination())
        l.plot_avg_loss()
    plt.plot(losslist)
    plt.show()

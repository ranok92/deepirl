import random
'''
file should contain methods and classes needed specifically to help run the rl methods
'''
import collections
import numpy as np
import pdb
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '..')


class LossBasedTermination():

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

    def update_loss_diff_list(self, new_loss):

        if self.last_loss is None:
            self.last_loss = new_loss

        else:
            new_diff = abs(self.last_loss - new_loss)
            self.loss_diff_list.append(new_diff)
            self.last_loss = new_loss
            self.current_avg_loss = sum(self.loss_diff_list)/self.list_size

        if self.current_avg_loss is not None:
            self.current_avg_loss_diff_list.append(self.current_avg_loss)

    def check_termination(self):
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

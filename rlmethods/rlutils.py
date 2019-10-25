'''General RL utilities.'''
import sys
import pdb
import random
import collections
from itertools import islice

from matplotlib import pyplot as plt
import numpy as np
sys.path.insert(0, '..')


class LossBasedTermination():

    def __init__(
            self,
            list_size=100,
            stop_threshold=.5,
            info=True,
            log_interval=50
    ):

        self.loss_diff_list = []
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

        elif len(self.loss_diff_list) == self.list_size:

            new_diff = abs(self.last_loss - new_loss)
            del(self.loss_diff_list[0])  # remove the oldest loss entry
            self.loss_diff_list.append(new_diff)
            self.last_loss = new_loss
            self.current_avg_loss = sum(self.loss_diff_list) / self.list_size

        else:

            new_diff = abs(self.last_loss - new_loss)
            self.loss_diff_list.append(new_diff)
            self.last_loss = new_loss

            if len(self.loss_diff_list) == self.list_size:
                self.current_avg_loss = sum(
                    self.loss_diff_list) / self.list_size

        if self.current_avg_loss is not None:
            if self.info:
                self.current_avg_loss_diff_list.append(self.current_avg_loss)

    def check_termination(self):

        if len(self.loss_diff_list) == self.list_size:

            if self.current_avg_loss < self.stop_threshold:

                return True

        return False

    def plot_avg_loss(self):

        if self.info:
            if len(self.current_avg_loss_diff_list) > 0 and len(self.current_avg_loss_diff_list) % self.log_interval == 0:
                plt.plot(self.current_avg_loss_diff_list)
                plt.draw()
                plt.pause(.0001)
        else:
            pass


class ReplayBuffer():
    """A general replay buffer for reinforcement learning replay."""

    def __init__(self, max_length):
        self.buffer = collections.deque(maxlen=max_length)

    def __len__(self):
        return len(self.buffer)

    def push(self, sars):
        """Push (s_t,a_t,r,s_(t+1)) transition tuple to replay buffer.

        :param sars: s(s_t,a_t,r,s_(t+1)) transition tuple.
        """
        self.buffer.append(sars)

    def sample(self, n):
        """Sample n samples from replay buffer.

        :param n: number of samples returned.
        """
        sample_batch = random.sample(self.buffer, n)
        inverted_batch = list(map(list, zip(*sample_batch)))
        sample_states = np.array(inverted_batch[0]).astype('float32')
        sample_actions = np.array(inverted_batch[1]).astype('float32')
        sample_rewards = np.array(inverted_batch[2]).astype('float32')
        sample_next_states = np.array(inverted_batch[3]).astype('float32')
        dones = np.array(np.array(inverted_batch[4])).astype('float32')

        return (
            sample_states,
            sample_actions,
            sample_rewards,
            sample_next_states,
            dones
        )

    def is_full(self):
        """returns true if replay buffer is full."""
        return len(self) == self.buffer.maxlen

class ContigousReplayBuffer(ReplayBuffer):
    """Replay buffer than samples n consecutive samples instead of randomly."""

    def sample(self, n):
        """Sample n contigous samples from replay buffer.

        :param n: number of samples returned.
        """
        assert len(self.buffer) >= n, "not enough samples in buffer."
        sample_idx = np.random.randint(0, len(self.buffer) - n)
        sample_batch = list(islice(self.buffer, sample_idx, sample_idx+n))

        inverted_batch = list(map(list, zip(*sample_batch)))
        sample_states = np.array(inverted_batch[0]).astype('float32')
        sample_actions = np.array(inverted_batch[1]).astype('float32')
        sample_rewards = np.array(inverted_batch[2]).astype('float32')
        sample_next_states = np.array(inverted_batch[3]).astype('float32')
        dones = np.array(np.array(inverted_batch[4])).astype('float32')

        return (
            sample_states,
            sample_actions,
            sample_rewards,
            sample_next_states,
            dones
        )


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

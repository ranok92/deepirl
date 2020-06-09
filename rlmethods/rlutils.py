'''General RL utilities.'''
import sys
import pdb
import random
import collections
from itertools import islice

from matplotlib import pyplot as plt
import numpy as np
import torch
sys.path.insert(0, '..')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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

def play(policy, env, feature_extractor, max_env_steps, render=False, best_action=True):
    """
    Plays the environment given the policy. use to render or debug environment/policy.

    :param policy: Policy to execute actions from.
    :type policy: Child of BasePolicy
    :param env: environment to play in.
    :type env: gym like environment.
    :param feature_extractor: feature extractor to use. Use identity feature
    extractor for gym environments.
    :type feature_extractor: Feature extractor like class.
    :param max_env_steps: maximum number of environment steps to take.
    :type max_env_steps: int.
    :param render: whether to render the environment or not., defaults to False
    :type render: bool, optional
    :param best_action: Whether to choose best (average) policy action or
    sample a random action according to policy., defaults to True
    :type best_action: bool, optional
    """
    done = False
    steps_counter = 0

    states = []
    features = []

    state = env.reset()
    states.append(state)
    state = feature_extractor.extract_features(state)
    state = torch.tensor(state).to(torch.float).to(DEVICE)
    features.append(state)

    if render:
        env.render()

    while not done and steps_counter < max_env_steps:

        if best_action:
            action = policy.eval_action(state)
        else:
            action, _, _ = policy.sample_action(state)

        next_state, _, done, _ = env.step(action)
        states.append(next_state)
        state = feature_extractor.extract_features(next_state)
        state = torch.tensor(state).to(torch.float).to(DEVICE)
        features.append(state)

        steps_counter += 1

        if render:
            env.render()

    return states, features

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

Transition = collections.namedtuple(
    "Transition",
    [
        "state",
        "action",
        "next_state",
        "reward",
        "done",
        "traj_end",
        "action_log_prob",
    ],
)
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)


def play_complete(policy, env, feature_extractor, max_steps, ped_id=None):
    """
    Plays using policy on environment for a maximum number of episodes.

    :param policy: Policy to use to play.
    :param env: Environment to play in.
    :param feature_extractor: feature extractor to use.
    :param max_episodes: Maximum number of steps to take.
    :return: Buffer of standard transition named tuples.

    """

    buffer = []
    done = False
    ep_length = 0

    state = feature_extractor.extract_features(env.reset(ped_id))

    while ep_length < max_steps:
        torch_state = torch.from_numpy(state).to(torch.float).to(DEVICE)

        action, log_prob, _ = policy.sample_action(torch_state)
        action = action.detach().cpu().numpy()

        log_prob = log_prob.detach().cpu().numpy()

        next_state, reward, done, _ = env.step(action)
        next_state = feature_extractor.extract_features(next_state)

        ep_length += 1
        max_steps_elapsed = ep_length >= max_steps

        buffer.append(
            Transition(
                state,
                action,
                next_state,
                reward,
                False if max_steps_elapsed else done,
                max_steps_elapsed,
                log_prob,
            )
        )

        state = next_state

        if done:
            break


    return buffer

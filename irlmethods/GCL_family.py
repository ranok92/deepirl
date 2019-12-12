""" Define GCL family IRL methods. """
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_nets.base_network import BaseNN, BasePolicy


class BaseExpert:
    """Base class for expert trajectory generation/retrieval."""

    def get_expert_trajectory(self):
        """Returns num_trajs number of trajectories as a list of tuples when
        called."""
        raise NotImplementedError


class PolicyExpert(BaseExpert):
    """Generates expert trajectores from an expert policy."""

    def __init__(self, policy, env, num_trajs, max_episode_length):
        if not isinstance(policy, BasePolicy):
            warnings.warn("Given policy is not a BasePolicy instance.")

        self.policy = policy
        self.env = env

        self.expert_trajs = self.generate_expert_trajectories(
            num_trajs, max_episode_length
        )

    def generate_expert_trajectories(self, num_trajs, max_episode_length=10000):
        """Generate a state buffer of expert trajectories.

        :param num_trajs: Number of trajectories to generate.
        :type num_trajs: int
        :param max_episode_length: max length of episodes, defaults to 10000
        :type max_episode_length: int, optional
        :return: list of (s, a, s', r, done) tuples
        :rtype: list of tuples.
        """
        buffer = []
        for _ in range(num_trajs):
            done = False
            state = self.env.reset()
            ep_length = 0
            while not (done or ep_length > max_episode_length):
                state = torch.from_numpy(state)
                action = self.policy.sample_action(state)
                action = action.detach().cpu().numpy()
                next_state, reward, done, _ = self.env.step(action)
                ep_length += 1

                if ep_length > max_episode_length:
                    buffer.append(state, action, next_state, reward, not done)
                else:
                    buffer.append(state, action, next_state, reward, done)

        return buffer

    def get_expert_trajectory(self):
        return self.expert_trajs


class RewardNetwork(BaseNN):
    """Simple 2 layer reward network."""

    def __init__(self, state_length, hidden_width):
        super().__init__()

        # define layers of NN
        self.linear1 = nn.Linear(state_length, hidden_width)
        self.linear2 = nn.Linear(hidden_width, hidden_width)
        self.head = nn.Linear(hidden_width, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.head(x))

        return x


class NaiveGCL:
    """Implements 'Guided Cost Learning' but without using policy optimizer
    used by Finn. et. al"""

    def __init__(self, rl_method, env, expert, reward_net=None):
        """Initialize a Naive version of GCL.
        
        :param rl_method: a class that does RL. 
        :type rl_method: Extension of BaseRL class.
        :param env: Gym-like environment.
        :type env: anyting that supports reset(), step() like gym.
        :param expert: expert that provides expert trajectories.
        :type expert: Extension of BaseExpert.
        :param reward_net: a reward network to train., defaults to None
        :type reward_net: Extension of BaseNN, optional
        """
        # RL related
        self.rl = rl_method
        self.env = env
        initial_state = env.reset()
        state_length = initial_state.shape[0]

        # IRL related
        self.expert = expert

        # NNs
        if not reward_net:
            self.reward = RewardNetwork(state_length, 256)

    def train(self):
        pass

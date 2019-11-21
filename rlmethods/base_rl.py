""" defines base RL class."""


class BaseRL:
    """Base RL class that all other RL methods should subclass."""

    def __init__(self):
        pass

    def train(self, num_episodes, max_episode_length, reward_network=None):
        """Train RL algorithm for num_episode number of episodes, and don't
        exceed max_episode_length when sampling from the environment.
        :param num_episodes: number of episodes to train RL for.
        :type num_episodes: int
        :param max_episode_length: maximum length of trajectories sampled
        from environment (if any.)
        :type max_episode_length: int
        :param reward_network: A reward network for obtaining rewards,
        defaults to None
        :type reward_network: pytorch policy, optional
        :return policy: trained policy.
        """
        raise NotImplementedError

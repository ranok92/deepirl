"""
Implements deep maxent IRL (Wulfmeier et. all) in a general, feature-type
agnostic way.
"""

import numpy as np
import torch
from torch.optim import Adam
from deep_maxent import RewardNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def play(env, policy, feature_extractor, max_env_steps):
    """
    Plays the environment using actions from supplied policy. Returns list of
    features encountered.

    :param env: gym-like environment (supporting step() and reset() functions.)

    :param policy: policy generating actions given features.
    :type policy: neural_nets.base_network.BaseRL

    :param feature_extractor: feature extractor which translates state
    dictionary to feature vector.
    :type feature_extractor: anything supporting
    extract_features(state_dictionary) function.

    :param max_env_steps: maximum allowed environment steps in each
    playthough (max length of rollout)
    :type max_env_steps: int

    :return: list of features encountered in playthrough.
    :rtype: list of 0D numpy arrays.
    """

    done = False
    steps_counter = 0
    states = []

    while not done and steps_counter < max_env_steps:
        action = policy.eval_action()
        action = action.cpu().numpy()

        state, _, done, _ = env.step(action)

        states.append(feature_extractor.extract_features(state))

        steps_counter += 1

    return states


class GeneralDeepMaxent:
    """
    Implements deep maxent IRL (Wulfmeier et. al) in a state-type agnostic way.
    """

    def __init__(self, rl, env, expert_states, learning_rate=1e-4):
        # environment attributes
        self.env = env
        state_size = env.reset().shape[0]

        # RL related
        self.rl = rl
        self.feature_extractor = self.rl.feature_extractor

        # reward net
        self.reward_net = RewardNet(state_size, hidden_dims=[256] * 2)
        self.reward_optim = Adam(
            self.reward_net.parameters(), lr=learning_rate
        )

        self.expert_states = expert_states

    def generate_trajectories(self, num_trajectories, max_env_steps):
        """
        Generate trajectories in environemnt using leanred RL policy.

        :param num_trajectories: number of trajectories to generate.
        :type num_trajectories: int

        :param max_env_steps: max steps to take in environment (rollout length.)
        :type max_env_steps: int

        :return: numpy array of features encountered in playthrough.
        :rtype: numpy array of shape (num_states x feature_length)
        """
        states = []

        for _ in range(num_trajectories):
            generated_states = play(
                self.env, self.rl.policy, self.feature_extractor, max_env_steps
            )

            states.extend(generated_states)

        return np.array(states)

    def train_episode(
        self,
        num_rl_episodes,
        max_rl_episode_length,
        num_trajectory_samples,
        max_env_steps,
    ):
        """
        perform IRL training.

        :param num_rl_episodes: Number of RL iterations for this IRL iteration.
        :type num_rl_episodes: int.

        :param max_rl_episode_length: maximum number of environment steps to
        take when doing rollouts using learned RL agent.
        :type max_rl_episode_length: int

        :param num_trajectory_samples: Number of trajectories to sample using
        learned RL agent.
        :type num_trajectory_samples: int

        :param max_env_steps: maximum number of environment steps to take,
        both when training RL agent and when generating rollouts.
        :type max_env_steps: int
        """

        # train RL agent
        self.rl.reset()
        self.rl.train(
            num_rl_episodes,
            max_rl_episode_length,
            reward_network=self.reward_net,
        )

        # expert loss
        torch_expert_states = torch.from_numpy(self.expert_states)
        torch_expert_states = torch_expert_states.to("float").to(DEVICE)
        expert_loss = self.reward_net(self.expert_states)

        # policy loss
        trajectories = self.generate_trajectories(
            num_trajectory_samples, max_env_steps
        )

        torch_trajs = torch.from_numpy(trajectories).to(torch.float).to(DEVICE)

        policy_loss = self.reward_net(torch_trajs).mean()

        # IRL step
        loss = policy_loss - expert_loss
        self.reward_optim.zero_grad()
        loss.backward()

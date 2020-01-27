"""
Implements deep maxent IRL (Wulfmeier et. all) in a general, feature-type
agnostic way.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tensorboardX import SummaryWriter

sys.path.insert(0, "..")
from neural_nets.base_network import BaseNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def play(env, policy, feature_extractor, max_env_steps, render=False):
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

    :param render: Render the environment. Defaults to False.
    :type render: Boolean.

    :return: list of features encountered in playthrough.
    :rtype: list of 0D numpy arrays.
    """

    done = False
    steps_counter = 0
    features = []

    feature = feature_extractor.extract_features(env.reset())
    torch_feature = torch.from_numpy(feature).to(torch.float).to(DEVICE)
    features.append(torch_feature)

    while not done and steps_counter < max_env_steps:
        action = policy.eval_action(torch_feature)

        # TODO: Fix misalignmnet in implementations of eval_action.
        # action = action.cpu().numpy()

        state, _, done, _ = env.step(action)

        if render:
            env.render()

        feature = feature_extractor.extract_features(state)
        torch_feature = torch.from_numpy(feature).to(torch.float).to(DEVICE)
        features.append(torch_feature)

        steps_counter += 1

    return features


class RewardNet(BaseNN):
    """Reward network"""

    def __init__(self, state_dims, hidden_dims=128):
        super(RewardNet, self).__init__()

        self.input = nn.Linear(state_dims, hidden_dims)

        self.linear1 = nn.Linear(hidden_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)

        self.head = nn.Linear(hidden_dims, 1)

    def forward(self, x):
        x = F.relu(self.input(x))

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        x = torch.tanh(self.head(x))

        return x


class GeneralDeepMaxent:
    """
    Implements deep maxent IRL (Wulfmeier et. al) in a state-type agnostic way.
    """

    def __init__(
        self,
        rl,
        env,
        expert_states,
        num_expert_trajs,
        learning_rate=1e-3,
        save_folder="./",
    ):
        # RL related
        self.rl = rl
        self.feature_extractor = self.rl.feature_extractor

        # environment attributes
        self.env = env
        state_size = self.feature_extractor.extract_features(
            env.reset()
        ).shape[0]

        # reward net
        self.reward_net = RewardNet(state_size, hidden_dims=256)
        self.reward_net = self.reward_net.to(DEVICE)
        self.reward_optim = Adam(
            self.reward_net.parameters(), lr=learning_rate, weight_decay=1e-5,
        )

        # expert info
        self.expert_states = expert_states.to(torch.float).to(DEVICE)
        self.num_expert_trajs = num_expert_trajs

        # logging and saving
        self.save_path = Path(save_folder)
        self.tbx_writer = SummaryWriter(
            str(self.save_path / "tensorboard_logs")
        )

        # training meta
        self.training_i = 0

    def generate_trajectories(self, num_trajectories, max_env_steps):
        """
        Generate trajectories in environemnt using leanred RL policy.

        :param num_trajectories: number of trajectories to generate.
        :type num_trajectories: int

        :param max_env_steps: max steps to take in environment (rollout length.)
        :type max_env_steps: int

        :return: list of features encountered in playthrough.
        :rtype: list of tensors of shape (num_states x feature_length)
        """
        states = []

        for _ in range(num_trajectories):
            generated_states = play(
                self.env, self.rl.policy, self.feature_extractor, max_env_steps
            )

            states.extend(generated_states)

        return states

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

        :param episode_i: Current IRL iteration count.
        :type episode_i: int
        """

        # train RL agent
        self.rl.reset_training()
        self.rl.train(
            num_rl_episodes,
            max_rl_episode_length,
            reward_network=self.reward_net,
        )

        # expert loss
        expert_loss = self.reward_net(self.expert_states).sum()

        # policy loss
        trajectories = self.generate_trajectories(
            num_trajectory_samples, max_env_steps
        )

        torch_trajs = torch.stack(trajectories).to(torch.float).to(DEVICE)

        policy_loss = self.reward_net(torch_trajs).sum()
        policy_loss = (
            self.num_expert_trajs / num_trajectory_samples
        ) * policy_loss

        # Backpropagate IRL loss
        loss = policy_loss - expert_loss

        self.reward_optim.zero_grad()
        loss.backward()
        self.reward_optim.step()

        # logging
        self.tbx_writer.add_scalar(
            "IRL/policy_loss", policy_loss, self.training_i
        )
        self.tbx_writer.add_scalar(
            "IRL/expert_loss", expert_loss, self.training_i
        )
        self.tbx_writer.add_scalar("IRL/total_loss", loss, self.training_i)

        # save policy and reward network
        # TODO: make a uniform dumping function for all agents.
        self.rl.policy.save(str(self.save_path / "policy"))
        self.reward_net.save(str(self.save_path / "reward_net"))

        # increment training counter
        self.training_i += 1

    def train(
        self,
        num_irl_episodes,
        num_rl_episodes,
        max_rl_episode_length,
        num_trajectory_samples,
        max_env_steps,
    ):
        for _ in range(num_irl_episodes):
            print("IRL episode {}".format(self.training_i))
            self.train_episode(
                num_rl_episodes,
                max_rl_episode_length,
                num_trajectory_samples,
                max_env_steps,
            )

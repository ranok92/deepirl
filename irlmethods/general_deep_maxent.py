"""
Implements deep maxent IRL (Wulfmeier et. all) in a general, feature-type
agnostic way.
"""
import sys
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tensorboardX import SummaryWriter

sys.path.insert(0, "..")
from neural_nets.base_network import BaseNN
from irlmethods.irlUtils import play_features as play
from rlmethods.rlutils import play_complete
import utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        expert_trajectories,
        learning_rate=1e-3,
        l2_regularization=1e-5,
        save_folder="./",
        saving_interval=10,
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
            self.reward_net.parameters(),
            lr=learning_rate,
            weight_decay=l2_regularization,
        )

        # expert info
        self.expert_trajectories = [
            traj.to(torch.float).to(DEVICE) for traj in expert_trajectories
        ]

        # logging and saving
        self.save_path = Path(save_folder)
        self.tbx_writer = SummaryWriter(
            str(self.save_path / "tensorboard_logs")
        )
        # highjack RL method's tbx_writer
        self.rl.tbx_writer = self.tbx_writer

        self.data_table = utils.DataTable()

        # training meta
        self.training_i = 0
        self.saving_interval = saving_interval

    def save_models(self, filename=None):
        self.rl.policy.save(str(self.save_path / "policy"), filename=filename)
        self.reward_net.save(
            str(self.save_path / "reward_net"), filename=filename
        )

    def generate_trajectories(
        self, num_trajectories, max_env_steps, stochastic,
    ):
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
                self.env,
                self.rl.policy,
                self.feature_extractor,
                max_env_steps,
                stochastic,
            )

            states.append(generated_states)

        return states

    def discounted_rewards(self, rewards, gamma, account_for_terminal_state):
        discounted_sum = 0
        t = 0
        gamma_t = 1
        for t, reward in enumerate(rewards[:-1]):
            discounted_sum += gamma_t * reward
            gamma_t *= gamma

        if account_for_terminal_state:
            discounted_sum += (
                (gamma / (1 - gamma)) * gamma ** (t + 1) * rewards[-1]
            )
        else:
            discounted_sum += gamma_t * rewards[-1]

        return discounted_sum

    def train_episode(
        self,
        num_rl_episodes,
        max_rl_episode_length,
        num_trajectory_samples,
        max_env_steps,
        reset_training,
        account_for_terminal_state,
        gamma,
        stochastic_sampling,
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

        :param reset_training: Whether to reset RL training every iteration
        or not.
        :type reset_training: Boolean.

        :param account_for_terminal_state: Whether to account for a state
        being terminal or not. If true, (gamma/1-gamma)*R will be immitated
        by padding the trajectory with its ending state until max_env_steps
        length is reached. e.g. if max_env_steps is 5, the trajectory [s_0,
        s_1, s_2] will be padded to [s_0, s_1, s_2, s_2, s_2].
        :type account_for_terminal_state: Boolean.

        :param gamma: The discounting factor.
        :type gamma: float.

        :param stochastic_sampling: Sample trajectories using stochastic
        policy instead of deterministic 'best action policy'
        :type stochastic_sampling: Boolean.
        """

        # expert loss
        expert_loss = 0
        for traj in self.expert_trajectories:
            expert_rewards = self.reward_net(traj)

            expert_loss += self.discounted_rewards(
                expert_rewards, gamma, account_for_terminal_state
            )

        # policy loss
        trajectories = self.generate_trajectories(
            num_trajectory_samples, max_env_steps, stochastic_sampling
        )

        policy_loss = 0
        for traj in trajectories:
            policy_rewards = self.reward_net(traj)
            policy_loss += self.discounted_rewards(
                policy_rewards, gamma, account_for_terminal_state
            )

        policy_loss = (
            len(self.expert_trajectories) / num_trajectory_samples
        ) * policy_loss

        # Backpropagate IRL loss
        loss = policy_loss - expert_loss

        self.reward_optim.zero_grad()
        loss.backward()
        self.reward_optim.step()

        # train RL agent
        if reset_training:
            self.rl.reset_training()

        self.rl.train(
            num_rl_episodes,
            max_rl_episode_length,
            reward_network=self.reward_net,
        )

        # logging
        self.tbx_writer.add_scalar(
            "IRL/policy_loss", policy_loss, self.training_i
        )
        self.tbx_writer.add_scalar(
            "IRL/expert_loss", expert_loss, self.training_i
        )
        self.tbx_writer.add_scalar("IRL/total_loss", loss, self.training_i)

        self.data_table.add_row(
            {
                "IRL/policy_loss": policy_loss.item(),
                "IRL/expert_loss": expert_loss.item(),
                "IRL/total_loss": loss.item(),
            },
            self.training_i,
        )

        # save policy and reward network
        # TODO: make a uniform dumping function for all agents.
        self.save_models(filename="{}.pt".format(self.training_i))

        # increment training counter
        self.training_i += 1

    def train(
        self,
        num_irl_episodes,
        num_rl_episodes,
        max_rl_episode_length,
        num_trajectory_samples,
        max_env_steps,
        reset_training=False,
        account_for_terminal_state=False,
        gamma=0.99,
        stochastic_sampling=False,
    ):
        """
        Runs the train_episode() function for 'num_irl_episodes' times. Other
        parameters are identical to the aforementioned function, with the same
        description and requirements.
        """
        for _ in range(num_irl_episodes):
            print("IRL episode {}".format(self.training_i), end="\r")
            self.train_episode(
                num_rl_episodes,
                max_rl_episode_length,
                num_trajectory_samples,
                max_env_steps,
                reset_training,
                account_for_terminal_state,
                gamma,
                stochastic_sampling,
            )


class MixingDeepMaxent(GeneralDeepMaxent):
    def __init__(
        self,
        rl,
        env,
        expert_trajectories,
        learning_rate=0.001,
        l2_regularization=1e-05,
        save_folder="./",
        saving_interval=25,
    ):
        super().__init__(
            rl,
            env,
            expert_trajectories,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            save_folder=save_folder,
            saving_interval=saving_interval,
        )

        # expert and training datasets
        self.all_trajectories = random.sample(
            expert_trajectories, len(expert_trajectories)
        )
        self.expert_label_trajectories = [
            traj.to(torch.float).to(DEVICE)
            for traj in self.all_trajectories[
                : len(self.all_trajectories) // 2
            ]
        ]
        self.expert_train_trajectories = [
            traj.to(torch.float).to(DEVICE)
            for traj in self.all_trajectories[
                len(self.all_trajectories) // 2 :
            ]
        ]

        self.pre_data_table = utils.DataTable()

    def train_episode(
        self,
        num_rl_episodes,
        max_rl_episode_length,
        max_env_steps,
        reset_training,
        account_for_terminal_state,
        gamma,
        stochastic_sampling,
        num_expert_samples,
        num_policy_samples,
    ):
        """
        perform IRL with mix-in of expert samples.

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

        :param reset_training: Whether to reset RL training every iteration
        or not.
        :type reset_training: Boolean.

        :param account_for_terminal_state: Whether to account for a state
        being terminal or not. If true, (gamma/1-gamma)*R will be immitated
        by padding the trajectory with its ending state until max_env_steps
        length is reached. e.g. if max_env_steps is 5, the trajectory [s_0,
        s_1, s_2] will be padded to [s_0, s_1, s_2, s_2, s_2].
        :type account_for_terminal_state: Boolean.

        :param gamma: The discounting factor.
        :type gamma: float.

        :param stochastic_sampling: Sample trajectories using stochastic
        policy instead of deterministic 'best action policy'
        :type stochastic_sampling: Boolean.
        """

        # expert loss
        expert_loss = 0
        expert_samples = random.sample(
            self.expert_trajectories, num_expert_samples
        )
        for traj in expert_samples:
            expert_rewards = self.reward_net(traj)

            expert_loss += self.discounted_rewards(
                expert_rewards, gamma, account_for_terminal_state
            )

        # policy loss
        trajectories = self.generate_trajectories(
            num_expert_samples // 2, max_env_steps, stochastic_sampling
        )
        # mix in expert samples.
        trajectories.extend(
            random.sample(self.expert_trajectories, num_policy_samples // 2)
        )

        policy_loss = 0
        for traj in trajectories:
            policy_rewards = self.reward_net(traj)
            policy_loss += self.discounted_rewards(
                policy_rewards, gamma, account_for_terminal_state
            )

        policy_loss = (num_expert_samples / num_policy_samples) * policy_loss

        # Backpropagate IRL loss
        loss = policy_loss - expert_loss

        self.reward_optim.zero_grad()
        loss.backward()
        self.reward_optim.step()

        # train RL agent
        if reset_training:
            self.rl.reset_training()

        self.rl.train(
            num_rl_episodes,
            max_rl_episode_length,
            reward_network=self.reward_net,
        )

        # logging
        self.tbx_writer.add_scalar(
            "IRL/policy_loss", policy_loss, self.training_i
        )
        self.tbx_writer.add_scalar(
            "IRL/expert_loss", expert_loss, self.training_i
        )
        self.tbx_writer.add_scalar("IRL/total_loss", loss, self.training_i)

        self.data_table.add_row(
            {
                "IRL/policy_loss": policy_loss.item(),
                "IRL/expert_loss": expert_loss.item(),
                "IRL/total_loss": loss.item(),
            },
            self.training_i,
        )

        # save policy and reward network
        # TODO: make a uniform dumping function for all agents.
        if self.training_i + 1 % self.saving_interval == 0:
            self.save_models(filename="{}.pt".format(self.training_i))

        # increment training counter
        self.training_i += 1

    def pre_train_episode(
        self, num_trajectory_samples, account_for_terminal_state, gamma,
    ):
        """
        perform IRL pre-training by using only expert samples.

        :param num_trajectory_samples: Number of trajectories to sample using
        learned RL agent.
        :type num_trajectory_samples: int

        :param account_for_terminal_state: Whether to account for a state
        being terminal or not. If true, (gamma/1-gamma)*R will be immitated
        by padding the trajectory with its ending state until max_env_steps
        length is reached. e.g. if max_env_steps is 5, the trajectory [s_0,
        s_1, s_2] will be padded to [s_0, s_1, s_2, s_2, s_2].
        :type account_for_terminal_state: Boolean.

        :param gamma: The discounting factor.
        :type gamma: float.
        """

        # expert loss
        expert_loss = 0
        expert_sample = random.sample(
            self.expert_label_trajectories, num_trajectory_samples
        )
        for traj in expert_sample:
            expert_rewards = self.reward_net(traj)

            expert_loss += self.discounted_rewards(
                expert_rewards, gamma, account_for_terminal_state
            )

        # policy loss
        trajectories = random.sample(
            self.expert_train_trajectories, num_trajectory_samples
        )

        generator_loss = 0
        for traj in trajectories:
            policy_rewards = self.reward_net(traj)
            generator_loss += self.discounted_rewards(
                policy_rewards, gamma, account_for_terminal_state
            )

        generator_loss = (
            len(self.expert_trajectories) / num_trajectory_samples
        ) * generator_loss

        # Backpropagate IRL loss
        loss = generator_loss - expert_loss

        self.reward_optim.zero_grad()
        loss.backward()
        self.reward_optim.step()

        # logging
        self.tbx_writer.add_scalar(
            "pre_IRL/generator_loss", generator_loss, self.training_i
        )
        self.tbx_writer.add_scalar(
            "pre_IRL/expert_loss", expert_loss, self.training_i
        )
        self.tbx_writer.add_scalar("pre_IRL/total_loss", loss, self.training_i)

        self.pre_data_table.add_row(
            {
                "pre_IRL/policy_loss": generator_loss.item(),
                "pre_IRL/expert_loss": expert_loss.item(),
                "pre_IRL/total_loss": loss.item(),
            },
            self.training_i,
        )

        # save policy and reward network
        self.reward_net.save(
            str(self.save_path / "reward_net"),
            filename="pre_{}.pt".format(self.training_i),
        )

        # increment training counter
        self.training_i += 1

    def pre_train(
        self,
        num_pretrain_episodes,
        num_trajectory_samples,
        account_for_terminal_state=False,
        gamma=0.99,
    ):
        """
        Runs the train_episode() function for 'num_irl_episodes' times. Other
        parameters are identical to the aforementioned function, with the same
        description and requirements.
        """
        for _ in range(num_pretrain_episodes):
            print(
                "IRL pre-training episode {}".format(self.training_i), end="\r"
            )
            self.pre_train_episode(
                num_trajectory_samples, account_for_terminal_state, gamma
            )

    def train(
        self,
        num_irl_episodes,
        num_rl_episodes,
        max_rl_episode_length,
        max_env_steps,
        reset_training=False,
        account_for_terminal_state=False,
        gamma=0.99,
        stochastic_sampling=False,
        num_expert_samples=64,
        num_policy_samples=64,
    ):
        """
        Runs the train_episode() function for 'num_irl_episodes' times. Other
        parameters are identical to the aforementioned function, with the same
        description and requirements.
        """
        for _ in range(num_irl_episodes):
            print("IRL episode {}".format(self.training_i), end="\r")
            self.train_episode(
                num_rl_episodes,
                max_rl_episode_length,
                max_env_steps,
                reset_training,
                account_for_terminal_state,
                gamma,
                stochastic_sampling,
                num_expert_samples,
                num_policy_samples,
            )


class GCL(MixingDeepMaxent):
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
        buffers = []

        for _ in range(num_trajectories):
            generated_buffer = play_complete(
                self.rl.policy,
                self.env,
                self.feature_extractor,
                max_env_steps,
            )

            buffers.append(generated_buffer)

        return buffers

    def train_episode(
        self,
        num_rl_episodes,
        max_rl_episode_length,
        max_env_steps,
        reset_training,
        account_for_terminal_state,
        gamma,
        stochastic_sampling,
        num_expert_samples,
        num_policy_samples,
    ):
        """
        perform IRL with mix-in of expert samples.

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

        :param reset_training: Whether to reset RL training every iteration
        or not.
        :type reset_training: Boolean.

        :param account_for_terminal_state: Whether to account for a state
        being terminal or not. If true, (gamma/1-gamma)*R will be immitated
        by padding the trajectory with its ending state until max_env_steps
        length is reached. e.g. if max_env_steps is 5, the trajectory [s_0,
        s_1, s_2] will be padded to [s_0, s_1, s_2, s_2, s_2].
        :type account_for_terminal_state: Boolean.

        :param gamma: The discounting factor.
        :type gamma: float.

        :param stochastic_sampling: Sample trajectories using stochastic
        policy instead of deterministic 'best action policy'
        :type stochastic_sampling: Boolean.
        """

        # expert loss
        expert_loss = 0
        expert_samples = random.sample(
            self.expert_trajectories, num_expert_samples
        )
        for traj in expert_samples:
            expert_rewards = self.reward_net(traj)

            expert_loss += self.discounted_rewards(
                expert_rewards, gamma, account_for_terminal_state
            )

        # policy loss
        trajectories = self.generate_trajectories(
            num_expert_samples // 2, max_env_steps
        )

        policy_loss = 0

        # mix in expert samples.
        expert_mixin_samples = random.sample(
            self.expert_trajectories, num_policy_samples // 2
        )

        for traj in expert_mixin_samples:
            policy_rewards = self.reward_net(traj)
            policy_loss += self.discounted_rewards(
                policy_rewards, gamma, account_for_terminal_state
            )

        # generator loss
        # approx Z from samples

        exponents = []
        traj_rewards = []

        for traj in trajectories:
            traj_states = [
                torch.from_numpy(tran.state).to(torch.float).to(DEVICE)
                for tran in traj
            ]
            traj_states = torch.stack(traj_states)

            rewards = self.reward_net(traj_states)
            traj_rewards.append(rewards.sum().clone())
            rewards = self.discounted_rewards(rewards, gamma, traj[-1].done)

            pi_log_probs = [
                torch.from_numpy(tran.action_log_prob)
                .to(torch.float)
                .to(DEVICE)
                for tran in traj
            ]

            exponent = rewards - torch.stack(pi_log_probs).sum()
            exponents.append(exponent)

        exponents = torch.cat(exponents)
        max_exponent = torch.max(exponents)

        log_Z = max_exponent + torch.exp(exponents - max_exponent).sum()
        print(log_Z)

        is_weights = torch.zeros(num_policy_samples // 2)
        for idx, traj in enumerate(trajectories):
            is_weights[idx] = torch.exp(exponents[idx] - log_Z)

        policy_loss += (torch.tensor(traj_rewards) * is_weights.detach()).sum()
        policy_loss = (num_expert_samples / num_policy_samples) * policy_loss

        # Backpropagate IRL loss
        loss = policy_loss - expert_loss

        self.reward_optim.zero_grad()
        loss.backward()
        self.reward_optim.step()

        # train RL agent
        if reset_training:
            self.rl.reset_training()

        self.rl.train(
            num_rl_episodes,
            max_rl_episode_length,
            reward_network=self.reward_net,
        )

        # logging
        self.tbx_writer.add_scalar(
            "IRL/policy_loss", policy_loss, self.training_i
        )
        self.tbx_writer.add_scalar(
            "IRL/expert_loss", expert_loss, self.training_i
        )
        self.tbx_writer.add_scalar("IRL/total_loss", loss, self.training_i)
        self.tbx_writer.add_scalar("IRL/log_Z", log_Z.item(), self.training_i)

        self.data_table.add_row(
            {
                "IRL/policy_loss": policy_loss.item(),
                "IRL/expert_loss": expert_loss.item(),
                "IRL/total_loss": loss.item(),
                "IRL/log_Z": log_Z.item(),
            },
            self.training_i,
        )

        # save policy and reward network
        # TODO: make a uniform dumping function for all agents.
        if self.training_i + 1 % self.saving_interval == 0:
            self.save_models(filename="{}.pt".format(self.training_i))

        # increment training counter
        self.training_i += 1


class ExpertOnlyMaxent:
    """
    Implements expert only deep maxent, using only expert demonstrations and
    no environment interaction.
    """

    def __init__(
        self,
        state_size,
        expert_trajectories,
        learning_rate=1e-3,
        l2_regularization=1e-5,
        save_folder="./",
    ):

        # reward net
        self.reward_net = RewardNet(state_size, hidden_dims=256)
        self.reward_net = self.reward_net.to(DEVICE)
        self.reward_optim = Adam(
            self.reward_net.parameters(),
            lr=learning_rate,
            weight_decay=l2_regularization,
        )

        # expert and training datasets
        self.all_trajectories = random.sample(
            expert_trajectories, len(expert_trajectories)
        )
        self.expert_trajectories = [
            traj.to(torch.float).to(DEVICE)
            for traj in self.all_trajectories[
                : len(self.all_trajectories) // 2
            ]
        ]
        self.training_trajectories = [
            traj.to(torch.float).to(DEVICE)
            for traj in self.all_trajectories[
                len(self.all_trajectories) // 2 :
            ]
        ]

        # logging and saving
        self.save_path = Path(save_folder)
        self.tbx_writer = SummaryWriter(
            str(self.save_path / "tensorboard_logs")
        )

        self.data_table = utils.DataTable()

        # training meta
        self.training_i = 0

    def discounted_rewards(self, rewards, gamma, account_for_terminal_state):
        discounted_sum = 0
        t = 0
        gamma_t = 1
        for t, reward in enumerate(rewards[:-1]):
            discounted_sum += gamma_t * reward
            gamma_t *= gamma

        if account_for_terminal_state:
            discounted_sum += (
                (gamma / (1 - gamma)) * gamma ** (t + 1) * rewards[-1]
            )
        else:
            discounted_sum += gamma_t * rewards[-1]

        return discounted_sum

    def train_episode(
        self, num_trajectory_samples, account_for_terminal_state, gamma,
    ):
        """
        perform IRL pre-training by using only expert samples.

        :param num_trajectory_samples: Number of trajectories to sample using
        learned RL agent.
        :type num_trajectory_samples: int

        :param account_for_terminal_state: Whether to account for a state
        being terminal or not. If true, (gamma/1-gamma)*R will be immitated
        by padding the trajectory with its ending state until max_env_steps
        length is reached. e.g. if max_env_steps is 5, the trajectory [s_0,
        s_1, s_2] will be padded to [s_0, s_1, s_2, s_2, s_2].
        :type account_for_terminal_state: Boolean.

        :param gamma: The discounting factor.
        :type gamma: float.
        """

        # expert loss
        expert_loss = 0
        expert_sample = random.sample(
            self.expert_trajectories, num_trajectory_samples
        )
        for traj in expert_sample:
            expert_rewards = self.reward_net(traj)

            expert_loss += self.discounted_rewards(
                expert_rewards, gamma, account_for_terminal_state
            )

        # policy loss
        trajectories = random.sample(
            self.training_trajectories, num_trajectory_samples
        )

        generator_loss = 0
        for traj in trajectories:
            policy_rewards = self.reward_net(traj)
            generator_loss += self.discounted_rewards(
                policy_rewards, gamma, account_for_terminal_state
            )

        generator_loss = (
            len(self.expert_trajectories) / num_trajectory_samples
        ) * generator_loss

        # Backpropagate IRL loss
        loss = generator_loss - expert_loss

        self.reward_optim.zero_grad()
        loss.backward()
        self.reward_optim.step()

        # logging
        self.tbx_writer.add_scalar(
            "IRL/generator_loss", generator_loss, self.training_i
        )
        self.tbx_writer.add_scalar(
            "IRL/expert_loss", expert_loss, self.training_i
        )
        self.tbx_writer.add_scalar("IRL/total_loss", loss, self.training_i)

        self.data_table.add_row(
            {
                "IRL/policy_loss": generator_loss.item(),
                "IRL/expert_loss": expert_loss.item(),
                "IRL/total_loss": loss.item(),
            },
            self.training_i,
        )

        # save policy and reward network
        self.reward_net.save(str(self.save_path / "reward_net"))

        # increment training counter
        self.training_i += 1

    def train(
        self,
        num_episodes,
        num_trajectory_samples,
        account_for_terminal_state=False,
        gamma=0.99,
    ):
        """
        Runs the train_episode() function for 'num_irl_episodes' times. Other
        parameters are identical to the aforementioned function, with the same
        description and requirements.
        """
        for _ in range(num_episodes):
            print(
                "IRL pre-training episode {}".format(self.training_i), end="\r"
            )
            self.train_episode(
                num_trajectory_samples, account_for_terminal_state, gamma
            )

import pdb
import argparse
import matplotlib
import numpy as np
import gym, sys, time, os
import glob
import torch
import numpy as np 
from matplotlib import pyplot as plt
sys.path.insert(0, '..')  # NOQA: E402
from logger.logger import Logger
import utils
from mountain_car import extract_features

from mountain_car import MCFeatures, MCFeaturesplain
import torch.multiprocessing as mp
from rlmethods.b_actor_critic import ActorCritic
import re

parser = argparse.ArgumentParser()
parser.add_argument('--policy-path', type=str, nargs='?', default=None)
parser.add_argument('--reward-path' , type=str, nargs='?', default= None)
parser.add_argument('--play', action='store_true',
                    help='play given or latest stored policy.')
parser.add_argument('--play-user' , action='store_true' , 
                    help='lets the user play the game')
parser.add_argument('--dont-save', action='store_true',
                   help="don't save the policy network weights.")
parser.add_argument('--render', action='store_true', help="show the env.")
parser.add_argument('--num-trajs', type=int, default=10)
parser.add_argument('--view-reward', action='store_true')
parser.add_argument('--policy-net-hidden-dims', nargs="*", type=int, default=[128])
parser.add_argument('--feat-extractor', type=str, default=None, help='The name of the \
                     feature extractor to be used in the experiment.')

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def main():
    
    args = parser.parse_args()

    experiment_logger = Logger('./results','temp_save.txt')


    experiment_logger.log_header('Arguments for the experiment :')
    experiment_logger.log_info(vars(args))
    
    mp.set_start_method('spawn')

    if args.render:
        from envs.gridworld import GridWorld
    else:
        from envs.gridworld_clockless import GridWorldClockless as GridWorld


    agent_width = 10
    step_size = 10
    obs_width = 10
    grid_size = 10


    if args.feat_extractor == 'Onehot':
        feat_ext = OneHot(grid_rows = 10 , grid_cols = 10)
    if args.feat_extractor == 'SocialNav':
        feat_ext = SocialNav(fieldList = ['agent_state','goal_state'])
    if args.feat_extractor == 'FrontBackSideSimple':
        feat_ext = FrontBackSideSimple(thresh1 = 1,
                                    thresh2 = 2,
                                    thresh3 = 3,
                                    thresh4=4,
                                    step_size=step_size,
                                    agent_width=agent_width,
                                    obs_width=obs_width,
                                    )

    if args.feat_extractor == 'LocalGlobal':
        feat_ext = LocalGlobal(window_size=3, grid_size=grid_size,
                           agent_width=agent_width, 
                           obs_width=obs_width,
                           step_size=step_size,
                           )
    
    feat_ext = MCFeatures(128, 128)
    experiment_logger.log_header('Parameters of the feature extractor :')
    experiment_logger.log_info(feat_ext.__dict__)

    '''
    np.asarray([2,2]),np.asarray([7,4]),np.asarray([3,5]),
                                np.asarray([5,2]),np.asarray([8,3]),np.asarray([7,5]),
                                np.asarray([3,3]),np.asarray([3,7]),np.asarray([5,7])
                                
    env = GridWorld(display=args.render, is_onehot= False,is_random=True,
                    rows=100, agent_width=agent_width,step_size=step_size,
                    obs_width=obs_width,width=grid_size,
                    cols=100,
                    seed=7,
                    buffer_from_obs=0,
                    obstacles=3,
                                
                    goal_state=np.asarray([5,5]))
    '''
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    experiment_logger.log_header('Environment details :')
    experiment_logger.log_info(env.__dict__)

    model = ActorCritic(env, feat_extractor=feat_ext,  gamma=0.99,
                        log_interval=100,max_ep_length=1000, hidden_dims=args.policy_net_hidden_dims,
                        max_episodes=3000)

    experiment_logger.log_header('Details of the RL method :')
    experiment_logger.log_info(model.__dict__)
    
    pdb.set_trace()

    if args.policy_path is not None:
        policy_file_list =  []
        reward_across_models = []
        if os.path.isfile(args.policy_path):
            policy_file_list.append(args.policy_path)
        if os.path.isdir(args.policy_path):
            policy_names = glob.glob(os.path.join(args.policy_path, '*.pt'))
            policy_file_list = sorted(policy_names, key=numericalSort)
        
        xaxis = np.arange(len(policy_file_list))

    if not args.play and not args.play_user:
        if args.reward_path is None:
            model.train_mp(n_jobs=4)
        else:
            from irlmethods.deep_maxent import RewardNet
            state_size = featExtract.extract_features(env.reset()).shape[0]
            reward_net = RewardNet(state_size)
            reward_net.load(args.reward_path)
            print(next(reward_net.parameters()).is_cuda)
            model.train_mp(reward_net = reward_net,n_jobs = 4)

        if not args.dont_save:  
            model.policy.save('./saved-models/')

    if args.play:
        for policy_file in policy_file_list:

            model.policy.load(policy_file)

            env.tickSpeed = 15
            assert args.policy_path is not None, 'pass a policy to play from!'

            reward_across_models.append(model.generate_trajectory(args.num_trajs, args.render))

        #plotting the 2d list

            xaxis = np.arange(len(reward_across_models))
            reward_across_models_np = np.array(reward_across_models)
            mean_rewards = np.mean(reward_across_models_np, axis=1)
            std_rewards = np.std(reward_across_models_np, axis=1)
            plt.plot(xaxis,mean_rewards,color = 'r',label='IRL trained agent')
            plt.fill_between(xaxis , mean_rewards-std_rewards , 
                        mean_rewards+std_rewards, alpha = 0.5, facecolor = 'r')
            plt.draw()
            plt.pause(0.001)
            #plt.show()
        plt.show()
    if args.play_user:
        env.tickSpeed = 200

        model.generate_trajectory_user(args.num_trajs, './trajs/ac_gridworld_user/')

if __name__ == '__main__':
    main()

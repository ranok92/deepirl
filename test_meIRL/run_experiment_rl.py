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

from mountain_car import MCFeatures, MCFeaturesplain, MCFeaturesOnehot
import torch.multiprocessing as mp
from rlmethods.b_actor_critic import ActorCritic

import re


import psutil
process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser()

parser.add_argument('--on-server', action='store_true')
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


parser.add_argument('--state-discretization', nargs="*", type=int, default=[128, 128], help='The number of discrete \
                    parts you want to break the state')
parser.add_argument('--save-folder', type=str, default=None, help='The name of the folder to store the results.')


numbers = re.compile(r'(\d+)')

def display_memory_usage(memory_in_bytes):

    units = ['B', 'KB', 'MB', 'GB', 'TB']
    mem_list = []
    cur_mem = memory_in_bytes
    while cur_mem > 1024:

        mem_list.append(cur_mem%1024)
        cur_mem /= 1024
    
    mem_list.append(cur_mem)
    for i in range(len(mem_list)):

        print(units[i] +':'+ str(mem_list[i])+', ', end='')

    print('\n')



def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def main():
    
    args = parser.parse_args()
    if args.on_server:
        # matplotlib without monitor
        matplotlib.use('Agg')

        # pygame without monitor
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    save_folder = './results/'+ args.save_folder
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

    if args.feat_extractor=='MCFeatures':
        feat_ext = MCFeatures(args.state_discretization[0], args.state_discretization[1]) 

    elif args.feat_extractor=='MCFeaturesOnehot':
        feat_ext = MCFeaturesOnehot(args.state_discretization[0], args.state_discretization[1])

    else:
        print('Enter proper feature extractor value.')
        exit()


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


    model = ActorCritic(env, feat_extractor=feat_ext,  gamma=0.99, plot_loss=True,
                        log_interval=10, max_ep_length=30, hidden_dims=args.policy_net_hidden_dims,
                        max_episodes=10, save_folder=save_folder)

    experiment_logger.log_header('Details of the RL method :')
    experiment_logger.log_info(model.__dict__)
    
    #pdb.set_trace()

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
            state_size = feat_ext.state_rep_size
            reward_net = RewardNet(state_size, args.policy_net_hidden_dims)
            reward_net.load(args.reward_path)
            print(next(reward_net.parameters()).is_cuda)
            model.train_mp(reward_net = reward_net,n_jobs = 4)

        if not args.dont_save:  
            model.policy.save('./saved-models/')

    if args.play:
        xaxis = []
        counter = 1

        for policy_file in policy_file_list:

            model.policy.load(policy_file)

            env.tickSpeed = 15
            assert args.policy_path is not None, 'pass a policy to play from!'

            reward_across_models.append(model.generate_trajectory(args.num_trajs, args.render))

        #plotting the 2d list

            xaxis.append(counter)
            counter += 1
            reward_across_models_np = np.array(reward_across_models)
            mean_rewards = np.mean(reward_across_models_np, axis=1)
            std_rewards = np.std(reward_across_models_np, axis=1)
            plt.plot(xaxis,mean_rewards,color = 'r',label='IRL trained agent')
            plt.fill_between(xaxis , mean_rewards-std_rewards , 
                        mean_rewards+std_rewards, alpha = 0.5, facecolor = 'r')
            plt.draw()
            plt.pause(0.001)
            '''
            print('RAM usage :')
            display_memory_usage(process.memory_info().rss)
            print('GPU usage :')
            display_memory_usage(torch.cuda.memory_allocated())
            torch.cuda.empty_cache()
            display_memory_usage(torch.cuda.memory_allocated())
            '''
            #plt.show()
        plt.show()
    if args.play_user:
        env.tickSpeed = 200

        model.generate_trajectory_user(args.num_trajs, './trajs/ac_gridworld_user/')

if __name__ == '__main__':
    main()

import pdb
import sys  # NOQA
sys.path.insert(0, '..')  # NOQA: E402

import numpy as np
import argparse
import torch.multiprocessing as mp
from rlmethods.b_actor_critic import ActorCritic

from featureExtractor.gridworld_featureExtractor import FrontBackSide,LocalGlobal,OneHot,SocialNav,FrontBackSideSimple

from utils import step_wrapper, reset_wrapper

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


parser.add_argument('--annotation-file', type=str, default='../envs/expert_datasets/data_zara/annotation/processed/crowds_zara01_processed.txt', 
                    help='The location of the annotation file to be used to run the environment.')
def main():
    
    args = parser.parse_args()
    mp.set_start_method('spawn')

    from envs.gridworld_drone import GridWorldDrone


    agent_width = 10
    step_size = 2
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
                                    fieldList = ['agent_state','goal_state','obstacles'])

    if args.feat_extractor == 'LocalGlobal':
        feat_ext = LocalGlobal(window_size=3, grid_size=grid_size,
                           agent_width=agent_width, 
                           obs_width=obs_width,
                           step_size=step_size,
                           fieldList = ['agent_state','goal_state','obstacles'])
    

    #featExtract = OneHot(grid_rows=10,grid_cols=10)
    #featExtract = FrontBackSideSimple(thresh1 = 1,fieldList =  ['agent_state','goal_state','obstacles'])

    #featExtract = SocialNav(fieldList = ['agent_state','goal_state'])
    '''
    np.asarray([2,2]),np.asarray([7,4]),np.asarray([3,5]),
                                np.asarray([5,2]),np.asarray([8,3]),np.asarray([7,5]),
                                np.asarray([3,3]),np.asarray([3,7]),np.asarray([5,7])
                               
    env = GridWorld(display=args.render, is_onehot= False,is_random=True,
                    rows=10, agent_width=agent_width,step_size=step_size,
                    obs_width=obs_width,width=grid_size,
                    cols=10,
                    seed = 7,
                    obstacles = '../envs/map3.jpg',
                                
                    goal_state = np.asarray([5,5]))
    '''


    env = GridWorldDrone(display=args.render, is_onehot = False, 
                        seed=999, obstacles=None, 
                        show_trail=False,
                        is_random=False,
                        annotation_file=args.annotation_file,
                        subject=None,
                        tick_speed=90, 
                        obs_width=10,
                        step_size=step_size,
                        agent_width=agent_width,
                        show_comparison=True,                       
                        rows=576, cols=720, width=grid_size)

    model = ActorCritic(env, feat_extractor=featExtract,  gamma=0.99,
                        log_interval=50,max_ep_length=500 , 
                        max_episodes = 2000)

    if args.policy_path is not None:
        model.policy.load(args.policy_path)
        
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
        #env.tickSpeed = 15
        assert args.policy_path is not None, 'pass a policy to play from!'

        model.generate_trajectory(args.num_trajs, './trajs/ac_loc_glob_rectified_win_3_static_map3/')

    if args.play_user:
        env.tickSpeed = 200

        model.generate_trajectory_user(args.num_trajs, './trajs/ac_gridworld_user/')

if __name__ == '__main__':
    main()

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

def main():
    
    args = parser.parse_args()
    mp.set_start_method('spawn')

    if args.render:
        from envs.gridworld import GridWorld
    else:
        from envs.gridworld_clockless import GridWorldClockless as GridWorld


    agent_width = 5
    step_size = 5
    obs_width = 5
    grid_size = 5
    '''
    featExtract = LocalGlobal(window_size=7, agent_width=agent_width,
                              step_size=step_size, 
                              obs_width=obs_width,
                              grid_size=grid_size,
                              fieldList = ['agent_state','goal_state','obstacles'])
    '''
    #featExtract = OneHot(grid_rows=10,grid_cols=10)
    #featExtract = FrontBackSideSimple(thresh1 = 1,fieldList =  ['agent_state','goal_state','obstacles'])

    #featExtract = SocialNav(fieldList = ['agent_state','goal_state'])
    feat_ext = FrontBackSideSimple(thresh1 = 1,
                                thresh2 = 2,
                                thresh3 = 3,
                                thresh4=4,
                                step_size=step_size,
                                agent_width=agent_width,
                                obs_width=obs_width,
                                fieldList = ['agent_state','goal_state','obstacles'])
    '''
    np.asarray([2,2]),np.asarray([7,4]),np.asarray([3,5]),
                                np.asarray([5,2]),np.asarray([8,3]),np.asarray([7,5]),
                                np.asarray([3,3]),np.asarray([3,7]),np.asarray([5,7])
                                '''
    env = GridWorld(display=args.render, is_onehot= False,is_random=True,
                    rows=60, agent_width=agent_width,step_size=step_size,
                    obs_width=obs_width,width=grid_size,
                    cols=60,
                    seed=7,
                    buffer_from_obs=0,
                    obstacles = [],
                                
                    goal_state = np.asarray([5,5]))

    model = ActorCritic(env, feat_extractor=feat_ext,  gamma=0.99,
                        log_interval=100,max_ep_length=100, hidden_dims=args.policy_net_hidden_dims,
                        max_episodes=4000)

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
        env.tickSpeed = 15
        assert args.policy_path is not None, 'pass a policy to play from!'

        model.generate_trajectory(args.num_trajs, './trajs/ac_fbs_simple4_static_map7/')

    if args.play_user:
        env.tickSpeed = 200

        model.generate_trajectory_user(args.num_trajs, './trajs/ac_gridworld_user/')

if __name__ == '__main__':
    main()

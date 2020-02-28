import numpy as np
import copy
import argparse
import torch
import pdb
import sys
import matplotlib.pyplot as plt
import pygame
from PIL import Image

sys.path.insert(0, '..')  # NOQA: E402
from featureExtractor.drone_feature_extractor import dist_2d, norm_2d
from featureExtractor.drone_feature_extractor import get_rot_matrix, deg_to_rad


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()



parser.add_argument('--feat-extractor', type=str, default=None)
parser.add_argument('--annotation-file', type=str, default='../envs/expert_datasets/\
university_students/annotation/processed/frame_skip_1/\
students003_processed_corrected.txt')

parser.add_argument('--reward-path', type=str, default=None,
                     help='Location of the reward file')

parser.add_argument('--reward-net-hidden-dims', nargs="*", default=[256], 
                    help='List containing the values of the hidden layers')


parser.add_argument('--policy-path', type=str, default=None,
                     help='Location of the policy file')

parser.add_argument('--policy-net-hidden-dims', nargs="*", default=[256], 
                    help='List containing the values of the hidden layers')


parser.add_argument('--frame-id', type=int, default=1500, 
                    help="The frame in the video to be used to calculate the reward map")

parser.add_argument('--sample-rate', type=int, nargs="*", default=[1, 1],
                    help='The gap in which to move the agent in the board to\
calculate the reward')

parser.add_argument('--render', action='store_true')



def generate_reward_map(env, feat_extractor, reward_network, 
                        render=False,
                        sample_rate=np.array([1, 1]), frame_id=10):
    """
    Given the environment, the current state and the sample rate place the 
    agent in the next spot in the environment to calculate the reward.
    input:
        environment    - The drone environment
        feat_extractor - The feature extractor in use to extract the features
                         This should be in accordance with the input
                         of the reward network
        reward_network - A pytorch network that calculates the reward 
                         given the state tensor.

        sample_rate    - Intervals in which the rewards are to be calculated.

        frame_id       - The still frame from the video on which the 
                         reward map is to be calculated


    output:
        reard_map      - a 2d numpy array containing the rewards obtained 
                         by the reward network for the given frame with
                         the provided sampling rate.

    """
    current_position = np.zeros(2) #[rows, cols]

    current_position = current_position.astype(int)
    rows = env.rows 
    cols = env.cols

    reward_map = np.zeros([int(rows/sample_rate[0])+1, int(cols/sample_rate[1])+1])

    row_counter = 0
    col_counter = 0

    env.current_frame = frame_id
    env.step()

    env.agent_state['position'] = current_position
    env.agent_state['speed'] = 0
    env.state['agent_state'] = copy.deepcopy(env.agent_state)
    
    env.render()

    env_display = env.gameDisplay

    data = pygame.image.tostring(env_display, 'RGBA')
    img = np.asarray(Image.frombytes('RGBA', (cols, rows), data))

    pygame.image.save(env_display, 'screencapture_fid'+str(frame_id)+'.png')
    while current_position[0] < rows:

        current_position[1] = 0
        col_counter = 0
        while current_position[1] < cols:

            env.agent_state['position'] = current_position
            env.agent_state['speed'] = 2
            env.agent_state['orientation'] = set_agent_orientation_to_goal(env.state)
            env.state['agent_state'] = copy.deepcopy(env.agent_state)
            if render:
                env.render()
            state_feat = feat_extractor.extract_features(env.state)
            state_feat = torch.from_numpy(state_feat).type(torch.FloatTensor).to(DEVICE)

            spot_reward = reward_network(state_feat)
            #pdb.set_trace()
            reward_map[row_counter, col_counter] = spot_reward
            col_counter += 1
            current_position[1] += sample_rate[1]

        current_position[0] += sample_rate[0]
        row_counter += 1

    return img, reward_map


def set_agent_orientation_to_goal(env_state):
    """
    Given the state of the environment, returns the orientation the agent should 
    have inorder to face directly towards the goal.

    input:
        state       - the current state of the environment
    

    output:
        orientation - A 2 dimensional numpy array containing the orientation (unit)vector
                      of the agent. The orientation vector is of the format [row, col]
    """
    agent_state = env_state['agent_state']
    goal_state = env_state['goal_state']

    vect = goal_state - agent_state['position']
    return vect/norm_2d(vect)








def plot_map(map_array, frame_img=None, colormap=None):
    """
    Plots and stores the plot of a heat map of the 2d array provided.
    input :
        map_array - A 2d numpy array containing the reward values obtained 
                    at a particular frame.
        frame_img - 


    """
    extent = 0, 720, 0, 576
    cmap=None
    if colormap is not None:
        cmap = plt.get_cmap('PiYG')
    fig, ax = plt.subplots()
    pdb.set_trace()

    if frame_img is not None:

        im1 = ax.imshow(frame_img, extent=extent, alpha=1)

    im = ax.imshow(map_array, cmap=cmap, alpha=0.5, extent=extent)
    fig.colorbar(im, ax=ax)

    # We want to show all ticks...
    #ax.set_xticks(np.arange(map_array.shape[1]))
    #ax.set_yticks(np.arange(map_array.shape[0]))
    # ... and label them with the respective list entries
    
    #ax.set_xticklabels(farmers)
    #ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #for i in range(map_array.shape[1]):
    #    for j in range(map_array.shape[0]):
    #        text = ax.text(j, i, map_array[j, i],
    #                       ha="center", va="center", color="w")

    ax.set_title("Reward plot")
    fig.tight_layout()
    plt.show()


def visualize_reward_per_spot(env, feat_extractor, reward_network, 
                              policy_network, num_traj=20, div=10,
                              render=True):

    """
    Runs the agent on the environment using the policy network passed and 
    displays the reward distribution the agent gets for all possible actions 
    at each step.

        input:
            env  - the environmnet
            feat_extractor - the feature extractor 
            reward_network - the reward network. A pytorch network.
            policy_network - The policy network the agent will follow. A pytorch network.
            render - flag set for rendering


        output : 
            N/A
    """

    for i in range(num_traj):
        
        done = False
        state = env.reset_and_replace()
        state_feat = feat_extractor.extract_features(state)
        state_feat = torch.from_numpy(state_feat).type(torch.FloatTensor).to(DEVICE)
        t = 0
        while not done:

            states_numpy = get_nearby_statevector(state, feat_extractor)
            states_torch = torch.from_numpy(states_numpy).type(torch.FloatTensor).to(DEVICE)

            rewards = reward_network(states_torch)
            rewards = rewards.detach().cpu().numpy().squeeze()
            rewards = (rewards-np.min(rewards))*3
            
            x_axis = np.linspace(0, 2*np.pi, int(360/div), endpoint=False)
            width = (2*np.pi)/int(360/div)
            ax = plt.subplot(111, polar=True)
            bars = ax.bar(x_axis, rewards, width=width, bottom=2)
            plt.draw()
            plt.pause(0.001)

            action = policy_network.eval_action(state_feat)
            state, _, done, _ = env.step(action)
            env.render()
            state_feat = feat_extractor.extract_features(state)
            state_feat = torch.from_numpy(state_feat).type(torch.FloatTensor).to(DEVICE)
            t+=1

            if t>600:
                break



def get_nearby_statevector(env_state, feat_extractor, div=10):

    """
    given the current state returns the rewards the agent would be getting if he would 
    be facing other directions
    """
    state_list = []
    orient_vector = np.asarray([-1, 0]) #starts facing upwards
    cur_orient_degree = 0
    for i in range(int(360/div)):
        env_state['agent_state']['orientation'] = orient_vector
        state_list.append(feat_extractor.extract_features(env_state))
        orient_vector = np.matmul(get_rot_matrix(deg_to_rad(cur_orient_degree+div)),
                                   orient_vector)
        cur_orient_degree += div

    return np.asarray(state_list)








def main():

    args = parser.parse_args()
    step_size = 2
    agent_width = 10
    obs_width = 10
    grid_size = 10

    #set up the feature extractor
    from featureExtractor.drone_feature_extractor import DroneFeatureRisk_speedv2
    from featureExtractor.drone_feature_extractor import VasquezF1, VasquezF2, VasquezF3

    feat_ext = None
    if args.feat_extractor == 'DroneFeatureRisk_speedv2':

        feat_ext = DroneFeatureRisk_speedv2(agent_width=agent_width,
                            obs_width=obs_width,
                            step_size=step_size,
                            grid_size=grid_size,
                            thresh1=18, thresh2=30)
    

    if args.feat_extractor == 'VasquezF1':
        feat_ext = VasquezF1(agent_width*6, 0.5, 1.0)

    if args.feat_extractor == 'VasquezF2':
        feat_ext = VasquezF1(agent_width*6, 0.5, 1.0)

    if args.feat_extractor == 'VasquezF3':
        feat_ext = VasquezF3(agent_width)

    #set up the environment
    from envs.gridworld_drone import GridWorldDrone

    env = GridWorldDrone(
        display=True,
        is_onehot=False,
        obstacles=None,
        show_trail=False,
        is_random=True,
        annotation_file=args.annotation_file,
        tick_speed=60,
        obs_width=10,
        step_size=step_size,
        agent_width=agent_width,
        replace_subject=False,
        consider_heading=True,
        show_orientation=True,
        rows=576,
        cols=720,
        width=grid_size,
    )

    #set up the policy network
    from rlmethods.b_actor_critic import Policy
    state_size = feat_ext.extract_features(env.reset()).shape[0]
    policy_net = Policy(state_size, env.action_space.n, args.policy_net_hidden_dims)
    policy_net.load(args.policy_path)
    print(next(policy_net.parameters()).is_cuda)


    #set up the reward network
    from irlmethods.deep_maxent import RewardNet

    state_size = feat_ext.extract_features(env.reset()).shape[0]
    reward_net = RewardNet(state_size, args.reward_net_hidden_dims)
    reward_net.load(args.reward_path)
    print(next(reward_net.parameters()).is_cuda)
    #run stuff
    '''
    screenshot, reward_map = generate_reward_map(env, feat_ext, 
                        reward_net, 
                        render=args.render,
                        sample_rate=args.sample_rate, 
                        frame_id=args.frame_id)

    plot_map(reward_map, frame_img=screenshot)
    '''

    visualize_reward_per_spot(env, feat_ext, reward_net, 
                              policy_net, num_traj=20,
                              render=True)

if __name__=='__main__':

    main()
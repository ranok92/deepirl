import numpy as np
import copy
import argparse
import torch
import pdb
import sys
import matplotlib.pyplot as plt



sys.path.insert(0, '..')  # NOQA: E402

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
    while current_position[0] < rows:

        current_position[1] = 0
        col_counter = 0
        while current_position[1] < cols:

            env.agent_state['position'] = current_position
            env.agent_state['speed'] = 2
            env.agent_state['orientation'] = np.asarray([1,1])
            env.state['agent_state'] = copy.deepcopy(env.agent_state)
            set_agent_orientation(env)
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

    pdb.set_trace()
    return reward_map


def set_agent_orientation(env):
    """
    Given the environment changes the orientation of the agennt so that it
    faces towards the goal.
    """


def plot_map(map_array, colormap=None):
    """
    Plots and stores the plot of a heat map of the 2d array provided.
    input :
        map_array - A 2d numpy array containing the reward values obtained 
                    at a particular frame.


    """
    cmap=None
    if colormap is not None:
        cmap = plt.get_cmap('PiYG')
    fig, ax = plt.subplots()

    im = ax.imshow(map_array, cmap=cmap)
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
        display=args.render,
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

    #set up the reward network
    from irlmethods.deep_maxent import RewardNet

    state_size = feat_ext.extract_features(env.reset()).shape[0]
    reward_net = RewardNet(state_size, args.reward_net_hidden_dims)
    reward_net.load(args.reward_path)
    print(next(reward_net.parameters()).is_cuda)
    #run stuff

    reward_map = generate_reward_map(env, feat_ext, 
                        reward_net, 
                        render=args.render,
                        sample_rate=args.sample_rate, 
                        frame_id=args.frame_id)

    plot_map(reward_map)

if __name__=='__main__':

    main()
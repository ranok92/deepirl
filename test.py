from debugtools import getperStateReward
from irlmethods.deep_maxent import RewardNet
import utils
from utils import to_oh
import itertools
import pdb
from envs.gridworld_clockless import GridWorldClockless as GridWorld
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


def per_state_reward(reward_function, rows, cols):
    all_states = itertools.product(range(rows), range(cols))

    oh_states = []
    for state in all_states:
        oh_states.append(utils.to_oh(state[0]*cols+state[1], rows*cols))

    all_states = torch.tensor(oh_states,
                              dtype=torch.float).to(DEVICE)

    return reward_function(all_states)

def main():
	r = 10
	c = 10
	env = GridWorld(display=False, obstacles=[np.asarray([1, 2])])
	reward_network = RewardNet(env.reset().shape[0])
	reward_network.load('./experiments/saved-models-rewards/1.pt')
	reward_network.eval()
	reward_network.to(DEVICE)
    
	reward_values = getperStateReward(reward_network, rows = 10 , cols = 10)
    
	irl_reward_valies = per_state_reward(reward_network,
        r,c)

	pdb.set_trace()

	plt.imshow(reward_values)
	plt.colorbar()
	plt.show()


if __name__ == '__main__':
        main()

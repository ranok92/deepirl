import sys
sys.path.insert(0, '..')
from rlmethods.b_actor_critic import ActorCritic
from envs.EWAP_gridworld import EwapGridworld
import torch

class DummyFeatureExtractor:
    def extract_features(self, state):
        torch_state = torch.from_numpy(state).type(torch.float).to('cuda')
        return torch_state

def main():
    fe = DummyFeatureExtractor()
    env = EwapGridworld(ped_id=1, vision_radius=4,)
    rl = ActorCritic(env, feat_extractor=fe, max_episodes=10**4)

    rl.train()

if __name__ == '__main__':
    main()

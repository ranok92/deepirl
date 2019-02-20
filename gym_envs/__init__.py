'''
register the modified forzenlake environment.
'''

from gym.envs.registration import register

register(id='FrozenLakeNP-v0',
         entry_point='gym_envs.np_frozenlake:FrozenLakeNP',
         kwargs={'map_name': '4x4'},
         max_episode_steps=100,
         reward_threshold=0.78,  # optimum = .8196
         )

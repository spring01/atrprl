
import os
os.environ['OMP_NUM_THREADS'] = '1'

import gym, atrp_ps

env = gym.make('ATRP-ps-v0')
state = env.reset()
while True:
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        break






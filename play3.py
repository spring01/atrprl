
import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
import gym, atrp_ps
import numpy as np

env = gym.make('ATRP-polystyrene-v5')

state = env.reset()
env.render()
total_reward = 0.0
for i in range(3000):
    state, reward, done, info = env.step(np.random.randint(env.action_space.n))
    total_reward += reward
    env.render()
    volume = '{:2.2f}'.format(env.unwrapped.volume)
    added = env.unwrapped.added
    print(i, reward, volume, round(added[0], 2), round(added[1], 2),
          round(added[2], 2), round(added[3], 2), round(added[4], 2))
    if done:
        break
env.render()
print('total reward:', total_reward)





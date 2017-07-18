
import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
import gym, atrp_ps_test
import numpy as np

env = gym.make('ATRP-polystyrene-test-v1')

state = env.reset()
env.render()
total_reward = 0.0
for i in range(3000):
    action = env.action_space.sample()
    action = np.random.randint(env.action_space.n)
    state, reward, done, info = env.step(action)
    total_reward += reward
    #~ env.render()
    volume = '{:2.2f}'.format(env.unwrapped.volume)
    quant_cu1 = env.unwrapped.quant[env.unwrapped.index[1]]
    quant_cu2 = env.unwrapped.quant[env.unwrapped.index[2]]
    added = env.unwrapped.added
    print(i, reward, volume, round(added[0], 3), round(added[1], 3),
          round(added[2], 2), round(added[3], 2), round(added[4], 2), round(quant_cu1, 2), round(quant_cu2, 2))
    if done:
        break
env.render()
print('total reward:', total_reward)





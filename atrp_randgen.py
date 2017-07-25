
import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
import gym, atrp_ps
import numpy as np
from hcdrl.common.envwrapper import HistoryStacker


env = gym.make('ATRP-ps-v0')
env = HistoryStacker(env, num_frames=1, act_steps=4)
state = env.reset()
env.render()
total_reward = 0.0
action_list = [(0, 0, 0, 0, 0),
               (1, 0, 0, 0, 0),
               (0, 1, 0, 0, 0),
               (0, 0, 1, 0, 0),
               (0, 0, 0, 1, 0),
               (0, 0, 0, 0, 1),]
for i in range(3000):
    action = action_list[np.random.randint(6)]
    state, reward, done, info = env.step(action)
    total_reward += reward
    volume = '{:2.2f}'.format(env.unwrapped.volume)
    quant_cu1 = env.unwrapped.quant[env.unwrapped.index[1]]
    quant_cu2 = env.unwrapped.quant[env.unwrapped.index[2]]
    added = env.unwrapped.added
    print(i, reward, volume, round(added[0], 3), round(added[1], 3),
          round(added[2], 2), round(added[3], 2), round(added[4], 2), round(quant_cu1, 2), round(quant_cu2, 2))
    if done:
        break
env.render()
chain = env.unwrapped.chain('dorm')
norm_chain = chain / np.sum(chain)
num_mono = np.arange(1, 101)
variance = norm_chain.dot(num_mono * num_mono) - norm_chain.dot(num_mono)**2
print(norm_chain)
print('variance:', variance)
print(np.argmax(norm_chain))




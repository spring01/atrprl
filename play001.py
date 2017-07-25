
import os
os.environ['OMP_NUM_THREADS'] = '1'
import gym
import numpy as np
from hcdrl.common.envwrapper import HistoryStacker
from gym.envs.registration import register
import atrp_ps_td_test


env = gym.make('ATRP-ps-td-var49-v0')
env = HistoryStacker(env, num_frames=1, act_steps=4)

for _ in range(100):
    state = env.reset()
    action_sequence = []
    num_steps = 0
    for i in range(3000):
        action = np.random.randint(6)
        action_sequence.append(action)
        state, reward, done, info = env.step(action)
        num_steps += 1
        if done:
            break
    env.render()
    print(reward)






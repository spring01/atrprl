
import os
os.environ['OMP_NUM_THREADS'] = '1'

import gym, atrp_ps_td_gv_ter
from hcdrl.common.envwrapper import HistoryStacker


env = gym.make('ATRP-ps-td-gv24-v0')
env = HistoryStacker(env, 1, 4)
state = env.reset()
act_seq = [5, 4, 4, 3, 2, 3, 5, 0, 2, 1, 4, 1, 4, 1, 1, 3, 5, 1, 3, 4, 4, 4, 2, 0, 1, 5, 0, 0, 4, 4, 3, 2, 4, 4, 4, 1, 0, 3, 4, 4, 2, 2, 4, 3, 1, 2, 4, 4, 5, 2, 5, 1, 5, 1, 4, 1, 1, 3, 3, 1, 2, 1, 1, 0, 1, 4, 1, 4, 5, 3, 4, 5, 1, 1, 2, 0, 1, 1, 1, 3, 2, 3, 5, 1, 3, 2, 1, 1, 0, 2]
for action in act_seq:
    state, reward, done, info = env.step(action)
env.render()








import os
os.environ['OMP_NUM_THREADS'] = '1'

import time

''' Tomek's values '''
N = 100
t_max = 1e6
kp = 1e4
ka = 2e-2
kd = 1e5
kt = 1e10
M_0 = 10.0
PBr1_0 = 0.4
CuBr_0 = 0.2
CuBr2_0 = CuBr_0 * 1.1


from atrp_env import ATRPEnv

env = ATRPEnv(timestep=1e6)
env.reset()
state = env.step(0)
print state







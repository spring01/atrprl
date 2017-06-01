
import os
os.environ['OMP_NUM_THREADS'] = '1'


from atrp_env import ATRPEnv

env = ATRPEnv(timestep=1e6)
state = env.reset()
state, reward, done, info = env.step(0)
env.render()
print state







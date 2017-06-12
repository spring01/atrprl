
import os
os.environ['OMP_NUM_THREADS'] = '1'


from atrp_env import ATRPEnv

env = ATRPEnv()
state = env.reset()
state, reward, done, info = env.step(env.action_space.sample())
env.render()
print state








import os
os.environ['OMP_NUM_THREADS'] = '1'

from atrp import ATRPEnv



env = ATRPEnv(max_rad_len=70, timestep=1e0, termination=False,
              k_prop=1.6e3, k_act=0.45, k_deact=1.1e7, k_ter=1e8,
              observation_mode='all', action_mode='single',
              mono_init=0.0, cu1_init=0.0, cu2_init=0.0, dorm1_init=0.0,
              mono_unit=0.1, cu1_unit=0.004, cu2_unit=0.004, dorm1_unit=0.008,
              mono_cap=10.0, cu1_cap=0.2, cu2_cap=0.2, dorm1_cap=0.4,
              mono_density=8.73,
              sol_init=0.01, sol_cap=0.0,
              cl_range=(20, 25))

state = env.reset()
#~ env.render()
total_reward = 0.0
for i in range(3000):
    state, reward, done, info = env.step(env.action_space.sample())
    total_reward += reward
    env.render()
    volume = '{:2.2f}'.format(env.volume)
    added = env.added
    print(i, reward, volume, round(added[0], 2), round(added[1], 2),
          round(added[2], 2), round(added[3], 2), round(added[4], 2))
    if done:
        break
#~ env.render()
print('total reward:', total_reward)

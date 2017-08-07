
import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import gym
from gym.envs.registration import register
from hcdrl.common.envwrapper import HistoryStacker


mean = 24
var = 40
space = np.linspace(1, 100, 100)
space_shifted = space - mean
target = np.exp(- space_shifted * space_shifted / (2 * var))
target /= np.sum(target)
register(
    id='ATRP-ps-td-test-v0',
    entry_point='atrp:ATRPTargetDistribution',
    max_episode_steps=100000,
    kwargs={
        'max_rad_len': 100,
        'step_time': 1e2,
        'completion_time': 1e5,
        'min_steps': 100,
        'termination': False,
        'k_prop': 1.6e3,
        'k_act': 0.45,
        'k_deact': 1.1e7,
        'k_ter': 1e8,
        'observation_mode': 'all stable',
        'mono_init': 0.0,
        'cu1_init': 0.0,
        'cu2_init': 0.0,
        'dorm1_init': 0.0,
        'mono_unit': 0.1,
        'cu1_unit': 0.004,
        'cu2_unit': 0.004,
        'dorm1_unit': 0.008,
        'mono_cap': 10.0,
        'cu1_cap': 0.2,
        'cu2_cap': 0.2,
        'dorm1_cap': 0.4,
        'mono_density': 8.73,
        'sol_init': 0.01,
        'sol_cap': 0.0,
        'reward_chain_type': 'dorm',
        'dn_distribution': target,
        'thres_loose': 5e-3,
        'thres_tight': 1e-3,
    }
)


env = gym.make('ATRP-ps-td-test-v0')
env = HistoryStacker(env, num_frames=1, act_steps=4)

for _ in range(100):
    state = env.reset()
    for i in range(3000):
        action = np.random.randint(6)
        state, reward, done, info = env.step(action)
        if done:
            break
    print(reward)
    env.render()

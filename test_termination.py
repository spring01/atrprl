
import os
os.environ['OMP_NUM_THREADS'] = '1'
import gym
import numpy as np
from hcdrl.common.envwrapper import HistoryStacker
from gym.envs.registration import register


simple = [2.33490714e-10,   4.99837917e-09,   5.35990968e-08,
          3.83925058e-07,   2.06683543e-06,   8.92124972e-06,
          3.21661993e-05,   9.96624268e-05,   2.70924458e-04,
          6.56546107e-04,   1.43633616e-03,   2.86593466e-03,
          5.25995419e-03,   8.94359320e-03,   1.41747625e-02,
          2.10520711e-02,   2.94347895e-02,   3.89033723e-02,
          4.87806533e-02,   5.82166846e-02,   6.63206416e-02,
          7.23086527e-02,   7.56320123e-02,   7.60569390e-02,
          7.36816262e-02,   6.88932401e-02,   6.22813075e-02,
          5.45311516e-02,   4.63209864e-02,   3.82405866e-02,
          3.07409463e-02,   2.41158845e-02,   1.85101882e-02,
          1.39454962e-02,   1.03545741e-02,   7.61611977e-03,
          5.58476948e-03,   4.11362656e-03,   3.06883202e-03,
          2.33715007e-03,   1.82825927e-03,   1.47357421e-03,
          1.22318946e-03,   1.04213641e-03,   9.06722023e-04,
          8.01361566e-04,   7.16058800e-04,   6.44526168e-04,
          5.82854811e-04,   5.28615650e-04,   4.80276081e-04,
          4.36835207e-04,   3.97603365e-04,   3.62072964e-04,
          3.29844912e-04,   3.00587680e-04,   2.74014885e-04,
          2.49873030e-04,   2.27934645e-04,   2.07994220e-04,
          1.89865536e-04,   1.73379709e-04,   1.58383583e-04,
          1.44738325e-04,   1.32318152e-04,   1.21009158e-04,
          1.10708225e-04,   1.01322027e-04,   9.27661100e-05,
          8.49640473e-05,   7.78466755e-05,   7.13513944e-05,
          6.54215334e-05,   6.00057778e-05,   5.50576500e-05,
          5.05350400e-05,   4.63997828e-05,   4.26172764e-05,
          3.91561374e-05,   3.59878914e-05,   3.30866924e-05,
          3.04290718e-05,   2.79937105e-05,   2.57612345e-05,
          2.37140300e-05,   2.18360775e-05,   2.01128008e-05,
          1.85309324e-05,   1.70783906e-05,   1.57441691e-05,
          1.45182376e-05,   1.33914515e-05,   1.23554703e-05,
          1.14026841e-05,   1.05261471e-05,   9.71951674e-06,
          8.97699966e-06,   8.29330192e-06,   7.66358426e-06,
          7.08342143e-06]
register(
    id='ATRP-ps-test-v0',
    entry_point='atrp:ATRPTargetDistrib',
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
        'observation_mode': 'all',
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
        'dn_distribution': simple,
        'ks_num_sample': 2e3,
    }
)

register(
    id='ATRP-ps-test-v1',
    entry_point='atrp:ATRPTargetDistrib',
    max_episode_steps=100000,
    kwargs={
        'max_rad_len': 100,
        'step_time': 1e2,
        'completion_time': 1e5,
        'min_steps': 100,
        'termination': True,
        'k_prop': 1.6e3,
        'k_act': 0.45,
        'k_deact': 1.1e7,
        'k_ter': 1e8,
        'observation_mode': 'all',
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
        'dn_distribution': simple,
        'ks_num_sample': 2e3,
    }
)


env = gym.make('ATRP-ps-test-v0')
env = HistoryStacker(env, num_frames=1, act_steps=4)
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
print(num_steps)
env.render()

env2 = gym.make('ATRP-ps-test-v1')
env2 = HistoryStacker(env2, num_frames=1, act_steps=4)
state = env2.reset()
num_steps = 0
for action in action_sequence:
    state, reward, done, info = env2.step(action)
    num_steps += 1
    if done:
        break
print(num_steps)
env2.render()





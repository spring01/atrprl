
import numpy as np
from gym.envs.registration import register


''' Two-level reward environments. +0.1 if close, +1.0 if very close. '''
entry_point = 'atrp:ATRPTargetDistribution'
max_episode_steps = 100000
kwargs_common = {
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
    'reward_loose': 0.1,
    'reward_tight': 1.0,
}

mean = 24
space = np.linspace(1, 100, 100)
space_shifted = space - mean
gv48 = np.exp(- space_shifted * space_shifted / (2 * 48))
gv48 /= np.sum(gv48)

kwargs_v1 = kwargs_common.copy()
kwargs_v1['dn_distribution'] = gv48
kwargs_v1['thres_loose'] = 1e-2
kwargs_v1['thres_tight'] = 3e-3
register(
    id='ATRP-ps-td-gv48-test3-v1',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_v1
)

kwargs_v2 = kwargs_common.copy()
kwargs_v2['dn_distribution'] = gv48
kwargs_v2['thres_loose'] = 5e-3
kwargs_v2['thres_tight'] = 3e-3
register(
    id='ATRP-ps-td-gv48-test3-v2',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_v2
)

kwargs_v3 = kwargs_common.copy()
kwargs_v3['dn_distribution'] = gv48
kwargs_v3['thres_loose'] = 1e-2
kwargs_v3['thres_tight'] = 1e-3
register(
    id='ATRP-ps-td-gv48-test3-v3',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_v3
)

kwargs_v3 = kwargs_common.copy()
kwargs_v3['dn_distribution'] = gv48
kwargs_v3['thres_loose'] = 5e-3
kwargs_v3['thres_tight'] = 1e-3
register(
    id='ATRP-ps-td-gv48-test3-v4',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_v3
)

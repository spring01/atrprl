
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
    'obs_mode': 'all stable',
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
    'thres_loose': 1e-2,
    'thres_tight': 3e-3,
}

mean = 24
space = np.linspace(1, 100, 100)
space_shifted = space - mean

gv24 = np.exp(- space_shifted * space_shifted / (2 * 24))
gv24 /= np.sum(gv24)
kwargs_gv24 = kwargs_common.copy()
kwargs_gv24['dn_distribution'] = gv24
register(
    id='ATRP-ps-td-gv24-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_gv24
)

gv28 = np.exp(- space_shifted * space_shifted / (2 * 28))
gv28 /= np.sum(gv28)
kwargs_gv28 = kwargs_common.copy()
kwargs_gv28['dn_distribution'] = gv28
register(
    id='ATRP-ps-td-gv28-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_gv28
)

gv32 = np.exp(- space_shifted * space_shifted / (2 * 32))
gv32 /= np.sum(gv32)
kwargs_gv32 = kwargs_common.copy()
kwargs_gv32['dn_distribution'] = gv32
register(
    id='ATRP-ps-td-gv32-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_gv32
)

gv36 = np.exp(- space_shifted * space_shifted / (2 * 36))
gv36 /= np.sum(gv36)
kwargs_gv36 = kwargs_common.copy()
kwargs_gv36['dn_distribution'] = gv36
register(
    id='ATRP-ps-td-gv36-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_gv36
)

gv40 = np.exp(- space_shifted * space_shifted / (2 * 40))
gv40 /= np.sum(gv40)
kwargs_gv40 = kwargs_common.copy()
kwargs_gv40['dn_distribution'] = gv40
register(
    id='ATRP-ps-td-gv40-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_gv40
)

gv44 = np.exp(- space_shifted * space_shifted / (2 * 44))
gv44 /= np.sum(gv44)
kwargs_gv44 = kwargs_common.copy()
kwargs_gv44['dn_distribution'] = gv44
register(
    id='ATRP-ps-td-gv44-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_gv44
)

gv48 = np.exp(- space_shifted * space_shifted / (2 * 48))
gv48 /= np.sum(gv48)
kwargs_gv48 = kwargs_common.copy()
kwargs_gv48['dn_distribution'] = gv48
register(
    id='ATRP-ps-td-gv48-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_gv48
)

gv52 = np.exp(- space_shifted * space_shifted / (2 * 52))
gv52 /= np.sum(gv52)
kwargs_gv52 = kwargs_common.copy()
kwargs_gv52['dn_distribution'] = gv52
register(
    id='ATRP-ps-td-gv52-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_gv52
)


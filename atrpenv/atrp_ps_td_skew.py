
import numpy as np
from scipy.special import erf
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

def skew_distribution(alpha, loc, scale):
    alpha_sq = alpha * alpha
    delta = alpha / np.sqrt(1 + alpha_sq)
    delta_sq = delta * delta
    mean = loc - delta * np.sqrt(scale * 2 / np.pi)
    var = scale / (1 - 2 * delta_sq / np.pi)
    space = np.linspace(1, 100, 100)
    space_shifted = (space - mean) / np.sqrt(var)
    gaussian = np.exp(- space_shifted * space_shifted / 2)
    skew = 1.0 + erf(alpha * space_shifted / np.sqrt(2.0))
    dist = gaussian * skew
    dist /= np.sum(dist)
    return dist


kwargs_m20 = kwargs_common.copy()
m20 = skew_distribution(alpha=-2.0, loc=25.0, scale=32.0)
kwargs_m20['dn_distribution'] = m20
register(
    id='ATRP-ps-td-skew-m20-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_m20
)

kwargs_m15 = kwargs_common.copy()
m15 = skew_distribution(alpha=-1.5, loc=25.0, scale=32.0)
kwargs_m15['dn_distribution'] = m15
register(
    id='ATRP-ps-td-skew-m15-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_m15
)

kwargs_m10 = kwargs_common.copy()
m10 = skew_distribution(alpha=-1.0, loc=24.0, scale=32.0)
kwargs_m10['dn_distribution'] = m10
register(
    id='ATRP-ps-td-skew-m10-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_m10
)

kwargs_m10_v1 = kwargs_common.copy()
m10_v1 = skew_distribution(alpha=-1.0, loc=24.0, scale=36.0)
kwargs_m10_v1['dn_distribution'] = m10_v1
register(
    id='ATRP-ps-td-skew-m10-v1',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_m10_v1
)

kwargs_m05 = kwargs_common.copy()
m05 = skew_distribution(alpha=-0.5, loc=24.0, scale=32.0)
kwargs_m05['dn_distribution'] = m05
register(
    id='ATRP-ps-td-skew-m05-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_m05
)

kwargs_p05 = kwargs_common.copy()
p05 = skew_distribution(alpha=0.5, loc=24.0, scale=32.0)
kwargs_p05['dn_distribution'] = p05
register(
    id='ATRP-ps-td-skew-p05-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_p05
)

kwargs_p10 = kwargs_common.copy()
p10 = skew_distribution(alpha=1.0, loc=24.0, scale=32.0)
kwargs_p10['dn_distribution'] = p10
register(
    id='ATRP-ps-td-skew-p10-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_p10
)

kwargs_p15 = kwargs_common.copy()
p15 = skew_distribution(alpha=1.5, loc=23.0, scale=32.0)
kwargs_p15['dn_distribution'] = p15
register(
    id='ATRP-ps-td-skew-p15-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_p15
)

kwargs_p20 = kwargs_common.copy()
p20 = skew_distribution(alpha=2.0, loc=23.0, scale=32.0)
kwargs_p20['dn_distribution'] = p20
register(
    id='ATRP-ps-td-skew-p20-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_p20
)

kwargs_p25 = kwargs_common.copy()
p25 = skew_distribution(alpha=2.5, loc=23.0, scale=32.0)
kwargs_p25['dn_distribution'] = p25
register(
    id='ATRP-ps-td-skew-p25-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_p25
)

kwargs_p30 = kwargs_common.copy()
p30 = skew_distribution(alpha=3.0, loc=23.0, scale=32.0)
kwargs_p30['dn_distribution'] = p30
register(
    id='ATRP-ps-td-skew-p30-v0',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_p30
)

kwargs_p30_v1 = kwargs_common.copy()
p30_v1 = skew_distribution(alpha=3.0, loc=23.0, scale=36.0)
kwargs_p30_v1['dn_distribution'] = p30_v1
register(
    id='ATRP-ps-td-skew-p30-v1',
    entry_point=entry_point,
    max_episode_steps=max_episode_steps,
    kwargs=kwargs_p30_v1
)



from atrp import ATRPEnv
from gym.envs.registration import register


register(
    id='ATRP-polystyrene-v0',
    entry_point='atrp:ATRPEnv',
    max_episode_steps=100000,
    kwargs={
        'max_rad_len': 70,
        'step_time': 1e1,
        'completion_time': 1e5,
        'min_steps': 100,
        'termination': False,
        'k_prop': 1.6e3,
        'k_act': 0.45,
        'k_deact': 1.1e7,
        'k_ter': 1e8,
        'observation_mode': 'all',
        'action_mode': 'single',
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
        'cl_range': (20, 25),
    }
)


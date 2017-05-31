
import gym
import numpy as np
from scipy.integrate import odeint


__all__ = ['ATRPEnv']

''' keys in a state '''
MONO = 'mono'
CU1 = 'cu1'
CU2 = 'cu2'
RAD = 'rad'
DORM = 'dorm'
TER = 'ter' # optional

class ATRPEnv(gym.Env):

    '''
    Length of `var` is `4 * (max radical chain length) + 2`.

    Partition of `var`:
        mono    = [M]                   = var[0],
        cu_i    = [CuBr]                = var[1],
        cu_ii   = [CuBr2]               = var[2],
        dorm    = [P1Br], ..., [PnBr]   = var[3:3+n],
        rad     = [P1.], ..., [Pn.]     = var[3+n:3+2*n],
        ter     = [T2], ..., [T2n]      = var[3+2*n:2+4*n] (optional).

    Other arguments (rate constants):
        k_poly: rate constant for (monomer consumption);
        k_act:  rate constant for (dormant chain --> radical);
        k_dorm: rate constant for (radical --> dormant chain);
        k_ter:  rate constant for (radical --> terminated chain).
    '''

    def __init__(self, timestep=1e1, max_rad_len=100,
                 k_poly=1e4, k_act=2e-2, k_dorm=1e5, k_ter=1e10,
                 termination=True,
                 mono_init=10.0, dorm_init=0.4, cu1_init=0.2, cu2_init=0.22):
        self.constants = k_poly, k_act, k_dorm, k_ter
        self.k_poly = k_poly
        self.k_act = k_act
        self.k_dorm = k_dorm
        self.k_ter = k_ter if termination else 0.0
        self.max_rad_len = max_rad_len
        self.termination = termination
        self.mono_init = mono_init
        self.cu1_init = cu1_init
        self.cu2_init = cu2_init
        self.dorm_init = dorm_init
        self.timestep = np.array([0.0, timestep])

        # variable indices (slices) for the ODE solver
        self.mono_idx = 0
        self.cu1_idx = 1
        self.cu2_idx = 2
        dorm_from = 3
        self.dorm_slice = slice(dorm_from, dorm_from + max_rad_len)
        rad_from = 3 + max_rad_len
        self.rad_slice = slice(rad_from, rad_from + max_rad_len)

        # slices for the 2 types of terminated species
        # type 1 is of length 2 to n
        ter1_from = 3 + 2 * max_rad_len
        self.ter1_slice = slice(ter1_from, ter1_from + max_rad_len - 1)

        # type 2 is of length n+1 to 2n
        ter2_from = 2 + 3 * max_rad_len
        self.ter2_slice = slice(ter2_from, ter2_from + max_rad_len)

        # total number of terminated species is 2n-1
        self.num_ter = 2 * max_rad_len - 1
        self.ter_slice = slice(ter1_from, ter1_from + self.num_ter)

    def _reset(self):
        self.state = {}
        self.state[MONO] = self.mono_init
        self.state[CU1] = self.cu1_init
        self.state[CU2] = self.cu2_init
        self.state[DORM] = np.zeros(self.max_rad_len)
        self.state[DORM][0] = self.dorm_init
        self.state[RAD] = np.zeros(self.max_rad_len)
        if self.termination:
            self.state[TER] = np.zeros(self.num_ter)
        return self.state

    def _step(self, action):
        mono = [self.state[MONO]]
        cu1 = [self.state[CU1]]
        cu2 = [self.state[CU2]]
        dorm = self.state[DORM]
        rad = self.state[RAD]
        var = np.concatenate([mono, cu1, cu2, dorm, rad])
        if self.termination:
            var = np.concatenate([var, self.state[TER]])
        sol = odeint(self._atrp_diff, var, self.timestep)[1]
        self.state = {}
        self.state[MONO] = sol[self.mono_idx]
        self.state[CU1] = sol[self.cu1_idx]
        self.state[CU2] = sol[self.cu2_idx]
        self.state[DORM] = sol[self.dorm_slice]
        self.state[RAD] = sol[self.rad_slice]
        if self.termination:
            self.state[TER] = sol[self.ter_slice]

        reward = 0.0
        done = False
        info = {}
        return self.state, reward, done, info

    def _atrp_diff(self, var, time):
        max_rad_len = self.max_rad_len
        mono_idx = self.mono_idx
        cu1_idx = self.cu1_idx
        cu2_idx = self.cu2_idx
        dorm_slice = self.dorm_slice
        rad_slice = self.rad_slice

        mono = var[mono_idx]
        cu1 = var[cu1_idx]
        cu2 = var[cu2_idx]
        dorm = var[dorm_slice]
        rad = var[rad_slice]

        dvar = np.zeros(len(var))

        kt2 = 2 * self.k_ter
        kp_mono = self.k_poly * mono
        kp_mono_rad = kp_mono * rad
        sum_rad = np.sum(rad)
        kp_mono_sum_rad = kp_mono * sum_rad

        # monomer
        dvar[mono_idx] = -kp_mono_sum_rad

        # dormant chains
        dvar_dorm = (self.k_dorm * cu2) * rad - (self.k_act * cu1) * dorm
        dvar[dorm_slice] = dvar_dorm

        # Cu(I)
        sum_dvar_dorm = np.sum(dvar_dorm)
        dvar[cu1_idx] = sum_dvar_dorm

        # Cu(II)
        dvar[cu2_idx] = -sum_dvar_dorm

        # radicals
        dvar_rad = - dvar_dorm - kp_mono_rad - (kt2 * sum_rad) * rad
        dvar_rad[1:] += kp_mono_rad[:-1]
        dvar[rad_slice] = dvar_rad

        # terminated chains
        if self.termination:
            # length 2 to n
            num_ter1 = max_rad_len - 1
            dvar_ter1 = np.zeros(num_ter1)
            for p in range(num_ter1):
                rad_part = rad[:(p + 1)]
                dvar_ter1[p] = rad_part.dot(rad_part[::-1])
            dvar[self.ter1_slice] = kt2 * dvar_ter1

            # length n+1 to 2n
            num_ter2 = max_rad_len
            dvar_ter2 = np.zeros(num_ter2)
            for p in range(num_ter2):
                rad_part = rad[p:]
                dvar_ter2[p] = rad_part.dot(rad_part[::-1])
            dvar[self.ter2_slice] = kt2 * dvar_ter2

        return dvar



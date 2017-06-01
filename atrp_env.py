
import gym
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


__all__ = ['ATRPEnv']

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

    metadata = {'render.modes': ['human']}

    def __init__(self, timestep=1e1, max_rad_len=100, termination=True,
                 k_poly=1e4, k_act=2e-2, k_dorm=1e5, k_ter=1e10,
                 mono_init=10.0, cu1_init=0.2, cu2_init=0.22, dorm_init=0.4,
                 mono_add=0.01, mono_cap=None, cu1_add=0.01, cu1_cap=None,
                 cu2_add=0.01, cu2_cap=None, dorm1_add=0.01, dorm1_cap=None):
        self.k_poly = k_poly
        self.k_act = k_act
        self.k_dorm = k_dorm
        self.k_ter = k_ter if termination else 0.0
        self.max_rad_len = max_rad_len
        self.termination = termination

        # variable indices (slices) for the ODE solver
        self.mono_idx = 0
        self.cu1_idx = 1
        self.cu2_idx = 2
        dorm_from = 3
        self.dorm_slice = slice(dorm_from, dorm_from + max_rad_len)
        rad_from = 3 + max_rad_len
        self.rad_slice = slice(rad_from, rad_from + max_rad_len)

        if termination:
            # slices for the 2 types of terminated chains (ter)
            # ter type 1: length 2 to n
            ter1_from = 3 + 2 * max_rad_len
            self.ter1_slice = slice(ter1_from, ter1_from + max_rad_len - 1)

            # ter type 2: length n+1 to 2n
            ter2_from = 2 + 3 * max_rad_len
            self.ter2_slice = slice(ter2_from, ter2_from + max_rad_len)

            # total number of terminated chains is 2n-1
            self.ter_slice = slice(ter1_from, ter1_from + 2 * max_rad_len - 1)

        # build initial variable and timestep
        state_len = 2 + 4 * max_rad_len if termination else 3 + 2 * max_rad_len
        self.var_init = np.zeros(state_len)
        self.var_init[self.mono_idx] = mono_init
        self.var_init[self.cu1_idx] = cu1_init
        self.var_init[self.cu2_idx] = cu2_init
        self.var_init[self.dorm_slice][0] = dorm_init
        self.timestep = np.array([0.0, timestep])

        # for rendering
        self.axes = None

    def _reset(self):
        self.mono_added = 0.0
        self.cu1_added = 0.0
        self.cu2_added = 0.0
        self.dorm1_added = 0.0
        self.state = self.var_init
        return self.state

    def _step(self, action):
        self.state = odeint(self.atrp_diff, self.state, self.timestep)[1]
        reward = 0.0
        done = False
        info = {}
        return self.state, reward, done, info

    def _render(self, mode='human', close=False):
        if close:
            if self.axes is not None:
                self.axes = None
                self.plots = None
                plt.close()
            return
        if self.axes is None:
            self.axes = {}
            self.plots = {}
            self.generate_plot('dorm')
            plt.title('Concentrations')
            self.generate_plot('rad')
            if self.termination:
                self.generate_plot('ter')
            plt.xlabel('Chain length')
            plt.tight_layout()

        self.update_plot('dorm')
        self.update_plot('rad')
        if self.termination:
            self.update_plot('ter')
        plt.draw()
        plt.pause(0.0001)

    def atrp_diff(self, var, time):
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
            for p in xrange(num_ter1):
                rad_part = rad[:(p + 1)]
                dvar_ter1[p] = rad_part.dot(rad_part[::-1])
            dvar[self.ter1_slice] = kt2 * dvar_ter1

            # length n+1 to 2n
            num_ter2 = max_rad_len
            dvar_ter2 = np.zeros(num_ter2)
            for p in xrange(num_ter2):
                rad_part = rad[p:]
                dvar_ter2[p] = rad_part.dot(rad_part[::-1])
            dvar[self.ter2_slice] = kt2 * dvar_ter2

        return dvar

    def generate_plot(self, key):
        if key == 'dorm':
            values = self.state[self.dorm_slice]
            space = np.linspace(1, len(values), len(values))
            num = 1
            label = 'Dormant chains'
        elif key == 'rad':
            values = self.state[self.rad_slice]
            space = np.linspace(1, len(values), len(values))
            num = 2
            label = 'Radical chains'
        elif key == 'ter':
            values = self.state[self.ter_slice]
            space = np.linspace(2, len(values) + 1, len(values))
            num = 3
            label = 'Terminated chains'
        axis = plt.subplot(3, 1, num)
        plot = axis.plot(space, values, label=label)[0]
        axis.legend()
        self.axes[key] = axis
        self.plots[key] = plot

    def update_plot(self, key):
        if key == 'dorm':
            values = self.state[self.dorm_slice]
        elif key == 'rad':
            values = self.state[self.rad_slice]
        elif key == 'ter':
            values = self.state[self.ter_slice]
        self.axes[key].set_ylim([0, np.max(values) * 1.1])
        self.plots[key].set_ydata(values)


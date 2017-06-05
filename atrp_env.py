
import gym
from gym import spaces
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


__all__ = ['ATRPEnv']

''' rate constants '''
K_POLY = 0
K_ACT  = 1
K_DORM = 2
K_TER  = 3

''' indices and actions '''
MONO  = 0
CU1   = 1
CU2   = 2
DORM1 = 3

''' chain types '''
DORM = 'dorm'
RAD  = 'rad'
TER  = 'ter'
TER_A = 'ter_a'
TER_B = 'ter_b'

''' epsilon for float comparison '''
EPS = 1e-2

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
                 mono_init=10.0, cu1_init=0.2, cu2_init=0.22, dorm1_init=0.4,
                 mono_unit=0.01, mono_cap=None, cu1_unit=0.01, cu1_cap=None,
                 cu2_unit=0.01, cu2_cap=None, dorm1_unit=0.01, dorm1_cap=None,
                 reward_type='chain length',
                 reward_chain_type='dorm', reward_range=(20, 30), reward_unit=0.01):
        self.rate_constant = {}
        self.rate_constant[K_POLY] = k_poly
        self.rate_constant[K_ACT] = k_act
        self.rate_constant[K_DORM] = k_dorm
        self.rate_constant[K_TER] = k_ter if termination else 0.0
        self.max_rad_len = max_rad_len
        self.termination = termination

        # variable indices (slices) for the ODE solver
        self.index = {}
        self.index[MONO] = 0
        self.index[CU1] = 1
        self.index[CU2] = 2
        self.index[DORM1] = 3
        dorm_from = 3
        self.index[DORM] = slice(dorm_from, dorm_from + max_rad_len)
        rad_from = 3 + max_rad_len
        self.index[RAD] = slice(rad_from, rad_from + max_rad_len)

        if termination:
            # slices for the 2 types of terminated chains (ter)
            # ter type 1: length 2 to n
            ter1_from = 3 + 2 * max_rad_len
            self.index[TER_A] = slice(ter1_from, ter1_from + max_rad_len - 1)

            # ter type 2: length n+1 to 2n
            ter2_from = 2 + 3 * max_rad_len
            self.index[TER_B] = slice(ter2_from, ter2_from + max_rad_len)

            # total number of terminated chains is 2n-1
            self.index[TER] = slice(ter1_from, ter1_from + 2 * max_rad_len - 1)

        # build initial variable and timestep
        state_len = 2 + 4 * max_rad_len if termination else 3 + 2 * max_rad_len
        self.observation_space = spaces.Box(0, np.inf, shape=(state_len,))
        self.var_init = np.zeros(state_len)
        self.var_init[self.index[MONO]] = mono_init
        self.var_init[self.index[CU1]] = cu1_init
        self.var_init[self.index[CU2]] = cu2_init
        self.var_init[self.index[DORM1]] = dorm1_init
        self.timestep = timestep
        self.ode_time = np.array([0.0, timestep])

        # actions
        action_tuple = tuple(spaces.Discrete(2) for _ in xrange(4))
        self.action_space = spaces.Tuple(action_tuple)
        self.add_unit = {}
        self.add_unit[MONO] = mono_unit
        self.add_unit[CU1] = cu1_unit
        self.add_unit[CU2] = cu2_unit
        self.add_unit[DORM1] = dorm1_unit
        self.add_cap = {}
        self.add_cap[MONO] = mono_cap
        self.add_cap[CU1] = cu1_cap
        self.add_cap[CU2] = cu2_cap
        self.add_cap[DORM1] = dorm1_cap

        # rewards
        self.reward_type = reward_type.lower()
        self.reward_chain_type = reward_chain_type.lower()
        self.reward_unit = reward_unit
        if self.reward_chain_type == 'dorm':
            start = 1
        elif self.reward_chain_type == 'ter':
            start = 2
        self.reward_slice = slice(*(r - start for r in reward_range))
        self.reward_chain_mono = np.arange(*reward_range)

        # rendering
        self.axes = None

    def _reset(self):
        self.added = {}
        self.added[MONO] = self.var_init[self.index[MONO]]
        self.added[CU1] = self.var_init[self.index[CU1]]
        self.added[CU2] = self.var_init[self.index[CU2]]
        self.added[DORM1] = self.var_init[self.index[DORM1]]
        self.state = self.var_init
        chain = self.state[self.index[self.reward_chain_type]]
        self.last_reward_chain = chain[self.reward_slice]
        return self.state

    def _step(self, action):
        old_state = self.state.copy()
        self._take_action(action)
        self.state = odeint(self._atrp_diff, self.state, self.ode_time)[1]
        done = self._done(old_state)
        if self.reward_type == 'chain length':
            reward = self._reward_chain_length()
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
            self._generate_plot(DORM)
            plt.title('Concentrations')
            self._generate_plot(RAD)
            if self.termination:
                self._generate_plot(TER)
            plt.xlabel('Chain length')
            plt.tight_layout()
        else:
            self._update_plot(DORM)
            self._update_plot(RAD)
            if self.termination:
                self._update_plot(TER)
        plt.draw()
        plt.pause(0.0001)

    def _take_action(self, action):
        self._add(action, MONO)
        self._add(action, CU1)
        self._add(action, CU2)
        self._add(action, DORM1)

    def _add(self, action, key):
        if action[key] and self._uncapped(key):
            unit = self.add_unit[key]
            self.state[self.index[key]] += unit
            self.added[key] += unit

    def _done(self, old_state):
        max_diff = np.max(np.abs(self.state - old_state))
        threshold = np.max(np.abs(self.state)) * EPS / self.timestep
        return max_diff < threshold and not self._uncapped(MONO)

    def _reward_chain_length(self):
        chain = self.state[self.index[self.reward_chain_type]]
        reward_chain = chain[self.reward_slice]
        diff_reward_chain = reward_chain - self.last_reward_chain
        diff_reward_chain_mono = diff_reward_chain * self.reward_chain_mono
        pos_reward = np.sum(diff_reward_chain_mono > self.reward_unit)
        neg_reward = np.sum(diff_reward_chain_mono < -self.reward_unit)
        if pos_reward or neg_reward:
            self.last_reward_chain = reward_chain
        return pos_reward - neg_reward

    def _uncapped(self, key):
        unit = self.add_unit[key]
        added_eps = self.added[key] + unit * EPS
        cap = self.add_cap[key]
        return cap is None or added_eps < cap

    def _atrp_diff(self, var, time):
        max_rad_len = self.max_rad_len

        k_poly = self.rate_constant[K_POLY]
        k_act = self.rate_constant[K_ACT]
        k_dorm = self.rate_constant[K_DORM]
        k_ter = self.rate_constant[K_TER]

        mono_index = self.index[MONO]
        cu1_index = self.index[CU1]
        cu2_index = self.index[CU2]
        dorm_slice = self.index[DORM]
        rad_slice = self.index[RAD]

        mono = var[mono_index]
        cu1 = var[cu1_index]
        cu2 = var[cu2_index]
        dorm = var[dorm_slice]
        rad = var[rad_slice]

        dvar = np.zeros(len(var))

        kt2 = 2 * k_ter
        kp_mono = k_poly * mono
        kp_mono_rad = kp_mono * rad
        sum_rad = np.sum(rad)
        kp_mono_sum_rad = kp_mono * sum_rad

        # monomer
        dvar[mono_index] = -kp_mono_sum_rad

        # dormant chains
        dvar_dorm = (k_dorm * cu2) * rad - (k_act * cu1) * dorm
        dvar[dorm_slice] = dvar_dorm

        # Cu(I)
        sum_dvar_dorm = np.sum(dvar_dorm)
        dvar[cu1_index] = sum_dvar_dorm

        # Cu(II)
        dvar[cu2_index] = -sum_dvar_dorm

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
            dvar[self.index[TER_A]] = kt2 * dvar_ter1

            # length n+1 to 2n
            num_ter2 = max_rad_len
            dvar_ter2 = np.zeros(num_ter2)
            for p in xrange(num_ter2):
                rad_part = rad[p:]
                dvar_ter2[p] = rad_part.dot(rad_part[::-1])
            dvar[self.index[TER_B]] = kt2 * dvar_ter2

        return dvar

    def _generate_plot(self, key):
        values = self.state[self.index[key]]
        if key == DORM:
            space = np.linspace(1, len(values), len(values))
            num = 1
            label = 'Dormant chains'
        elif key == RAD:
            space = np.linspace(1, len(values), len(values))
            num = 2
            label = 'Radical chains'
        elif key == TER:
            space = np.linspace(2, len(values) + 1, len(values))
            num = 3
            label = 'Terminated chains'
        axis = plt.subplot(3, 1, num)
        plot = axis.plot(space, values, label=label)[0]
        axis.legend()
        self.axes[key] = axis
        self.plots[key] = plot

    def _update_plot(self, key):
        values = self.state[self.index[key]]
        self.axes[key].set_ylim([0, np.max(values) * 1.1])
        self.plots[key].set_ydata(values)


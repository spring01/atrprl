
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
SOL   = 4

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
    Length of `quant` is `4 * (max radical chain length) + 2`.

    Partition of `quant`:
        mono  = [M]                 = quant[0],
        cu_i  = [CuBr]              = quant[1],
        cu_ii = [CuBr2]             = quant[2],
        dorm  = [P1Br], ..., [PnBr] = quant[3:3+n],
        rad   = [P1.], ..., [Pn.]   = quant[3+n:3+2*n],
        ter   = [T2], ..., [T2n]    = quant[3+2*n:2+4*n] (optional).

    Rate constants:
        k_poly: rate constant for (monomer consumption);
        k_act:  rate constant for (dormant chain --> radical);
        k_dorm: rate constant for (radical --> dormant chain);
        k_ter:  rate constant for (radical --> terminated chain).

    Action related:
        mono_init:    initial quantity of monomer;
        mono_density: density of monomer (useful in calculating the volume);
        mono_unit:    unit amount of the "adding monomer" action;
        mono_cap:     maximum quantity (budget) of monomer.
        (Other X_init, X_unit, X_cap variables have similar definitions.)

    Reward related:
        reward_mode:       'chain length' (cl) or 'distribution' (dn);
        reward_chain_type: type of chain that the reward is related with.
    'chain length' mode:
        cl_range: range of desired chain lengths
                  (left inclusive, right exclusive);
        cl_unit:  unit of change in equivalent amount of monomer
                  considered as rewarding.
    'distribution' mode:
        dn_dist:  desired distribution (of the rewarded chain type).

    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, timestep=1e1, max_rad_len=100, termination=True,
                 k_poly=1e4, k_act=2e-2, k_dorm=1e5, k_ter=1e10,
                 mono_init=9.0, mono_density=9.0, mono_unit=0.01, mono_cap=None,
                 cu1_init=0.2, cu1_unit=0.01, cu1_cap=None,
                 cu2_init=0.2, cu2_unit=0.01, cu2_cap=None,
                 dorm1_init=0.4, dorm1_unit=0.01, dorm1_cap=None,
                 sol_init=0.0, sol_density=1.0, sol_unit=0.01, sol_cap=0.0,
                 reward_mode='chain length', reward_chain_type='dorm',
                 cl_range=(20, 30), cl_unit=0.01,
                 dn_dist=None):
        rate_constant = {K_POLY: k_poly, K_ACT: k_act, K_DORM: k_dorm}
        rate_constant[K_TER] = k_ter if termination else 0.0
        self.rate_constant = rate_constant
        self.max_rad_len = max_rad_len
        self.termination = termination

        # variable indices (slices) for the ODE solver
        index = {MONO: 0, CU1: 1, CU2: 2, DORM1: 3}
        dorm_from = 3
        index[DORM] = slice(dorm_from, dorm_from + max_rad_len)
        rad_from = 3 + max_rad_len
        index[RAD] = slice(rad_from, rad_from + max_rad_len)

        if termination:
            # slices for the 2 types of terminated chains (ter)
            # ter type 1: length 2 to n
            ter1_from = 3 + 2 * max_rad_len
            index[TER_A] = slice(ter1_from, ter1_from + max_rad_len - 1)

            # ter type 2: length n+1 to 2n
            ter2_from = 2 + 3 * max_rad_len
            index[TER_B] = slice(ter2_from, ter2_from + max_rad_len)

            # total number of terminated chains is 2n-1
            index[TER] = slice(ter1_from, ter1_from + 2 * max_rad_len - 1)
        self.index = index

        # build initial variable and timestep
        state_len = 2 + 4 * max_rad_len if termination else 3 + 2 * max_rad_len
        self.observation_space = spaces.Box(0, np.inf, shape=(state_len,))
        self.init_amount = {MONO: mono_init, CU1: cu1_init, CU2: cu2_init,
                            DORM1: dorm1_init, SOL: sol_init}
        self.density = {MONO: mono_density, SOL: sol_density}
        self.volume = volume = mono_init / mono_density + sol_init / sol_density

        quant_init = np.zeros(state_len)
        quant_init[index[MONO]] = mono_init
        quant_init[index[CU1]] = cu1_init
        quant_init[index[CU2]] = cu2_init
        quant_init[index[DORM1]] = dorm1_init
        self.quant_init = quant_init
        self.timestep = timestep
        self.ode_time = np.array([0.0, timestep])

        # actions
        action_tuple = tuple(spaces.Discrete(2) for _ in range(5))
        self.action_space = spaces.Tuple(action_tuple)
        self.add_unit = {MONO: mono_unit, CU1: cu1_unit, CU2: cu2_unit,
                         DORM1: dorm1_unit, SOL: sol_unit}
        self.add_cap = {MONO: mono_cap, CU1: cu1_cap, CU2: cu2_cap,
                        DORM1: dorm1_cap, SOL: sol_cap}
        self.volume_unit = {MONO: mono_unit / mono_density,
                            SOL: sol_unit / sol_density}

        # rewards
        self.reward_mode = reward_mode.lower()
        reward_chain_type = reward_chain_type.lower()
        self.reward_chain_type = reward_chain_type
        if self.reward_mode == 'chain length':
            reward_chain_type = reward_chain_type.lower()
            self.cl_unit = cl_unit
            if reward_chain_type == DORM:
                start = 1
            elif reward_chain_type == TER:
                start = 2
            self.cl_slice = slice(*(r - start for r in cl_range))
            self.cl_num_mono = np.arange(*cl_range)
        elif self.reward_mode == 'distribution':
            if dn_dist is None:
                chain_slice = index[reward_chain_type]
                dn_dist = np.ones(chain_slice.stop - chain_slice.start)
                dn_dist /= np.sum(dn_dist)
            dn_dist = dn_dist.copy()
            dn_dist_eps = min(1e-16, np.min(dn_dist[dn_dist > 0]))
            self.dn_dist_eps = dn_dist[dn_dist == 0] = dn_dist_eps
            dn_dist /= np.sum(dn_dist)
            self.dn_dist = dn_dist

        # rendering
        self.axes = None

    def _reset(self):
        self.added = self.init_amount.copy()
        self.quant = self.quant_init
        if self.reward_mode == 'chain length':
            chain = self.quant[self.index[self.reward_chain_type]]
            self.last_reward_chain = chain[self.cl_slice]
        return self.quant

    def _step(self, action):
        old_quant = self.quant.copy()
        self._take_action(action)
        conc = self.quant / self.volume
        conc = odeint(self._atrp_diff, conc, self.ode_time,
                      Dfun=self._atrp_diff_jac)[1]
        self.quant = conc * self.volume
        done = self._done(old_quant)
        if self.reward_mode == 'chain length':
            reward = self._reward_chain_length()
        elif self.reward_mode == 'distribution':
            reward = self._reward_distribution()
        info = {}
        return self.quant, reward, done, info

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
            plt.title('Quantities')
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
        self._add(action, MONO, change_volume=True)
        self._add(action, CU1)
        self._add(action, CU2)
        self._add(action, DORM1)
        self._add_sol(action)

    def _add(self, action, key, change_volume=False):
        if action[key] and self._uncapped(key):
            unit = self.add_unit[key]
            self.quant[self.index[key]] += unit
            if change_volume:
                self.volume += self.volume_unit[key]
            self.added[key] += unit

    def _add_sol(self, action):
        if action[SOL] and self._uncapped(SOL):
            self.volume += self.volume_unit[SOL]
            self.added[SOL] += self.add_unit[SOL]

    def _done(self, old_quant):
        max_diff = np.max(np.abs(self.quant - old_quant))
        threshold = np.max(np.abs(self.quant)) * EPS / self.timestep
        capped = not self._uncapped(MONO) and not self._uncapped(SOL)
        return max_diff < threshold and capped

    def _reward_chain_length(self):
        chain = self.quant[self.index[self.reward_chain_type]]
        reward_chain = chain[self.cl_slice]
        diff_reward_chain = reward_chain - self.last_reward_chain
        diff_cl_num_mono = diff_reward_chain * self.cl_num_mono
        pos_reward = np.sum(diff_cl_num_mono > self.cl_unit)
        neg_reward = np.sum(diff_cl_num_mono < -self.cl_unit)
        if pos_reward or neg_reward:
            self.last_reward_chain = reward_chain
        return pos_reward - neg_reward

    def _reward_distribution(self):
        chain = self.quant[self.index[self.reward_chain_type]]
        if np.sum(chain) <= 0.0:
            chain = np.ones(len(chain))
        curr_dist = chain / np.sum(chain)
        min_curr_dist = np.min(curr_dist[curr_dist > 0])
        min_eps = min(self.dn_dist_eps, min_curr_dist)
        curr_dist[curr_dist <= 0] = min_eps
        curr_dist = curr_dist / np.sum(curr_dist)
        kl_div = curr_dist.dot(np.log(curr_dist / self.dn_dist))
        return -kl_div

    def _uncapped(self, key):
        unit = self.add_unit[key]
        added_eps = self.added[key] + unit * EPS
        cap = self.add_cap[key]
        return cap is None or added_eps < cap

    def _atrp_diff(self, var, time):
        max_rad_len = self.max_rad_len

        rate_constant = self.rate_constant
        k_poly = rate_constant[K_POLY]
        k_act = rate_constant[K_ACT]
        k_dorm = rate_constant[K_DORM]
        k_ter = rate_constant[K_TER]

        index = self.index
        mono_index = index[MONO]
        cu1_index = index[CU1]
        cu2_index = index[CU2]
        dorm_slice = index[DORM]
        rad_slice = index[RAD]

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
        dvar_dorm = dvar[dorm_slice]
        dvar_dorm[:] = (k_dorm * cu2) * rad - (k_act * cu1) * dorm

        # Cu(I)
        sum_dvar_dorm = np.sum(dvar_dorm)
        dvar[cu1_index] = sum_dvar_dorm

        # Cu(II)
        dvar[cu2_index] = -sum_dvar_dorm

        # radicals
        dvar_rad = dvar[rad_slice]
        dvar_rad[:] = -dvar_dorm - kp_mono_rad - (kt2 * sum_rad) * rad
        dvar_rad[1:] += kp_mono_rad[:-1]

        # terminated chains
        if self.termination:
            # length 2 to n
            dvar_ter1 = dvar[index[TER_A]]
            for p in range(max_rad_len - 1):
                rad_part = rad[:(p + 1)]
                dvar_ter1[p] = rad_part.dot(rad_part[::-1])
            dvar_ter1 *= kt2

            # length n+1 to 2n
            dvar_ter2 = dvar[index[TER_B]]
            for p in range(max_rad_len):
                rad_part = rad[p:]
                dvar_ter2[p] = rad_part.dot(rad_part[::-1])
            dvar_ter2 *= kt2

        return dvar

    def _atrp_diff_jac(self, var, time):
        max_rad_len = self.max_rad_len

        rate_constant = self.rate_constant
        k_poly = rate_constant[K_POLY]
        k_act = rate_constant[K_ACT]
        k_dorm = rate_constant[K_DORM]
        k_ter = rate_constant[K_TER]

        index = self.index
        mono_index = index[MONO]
        cu1_index = index[CU1]
        cu2_index = index[CU2]
        dorm_slice = index[DORM]
        rad_slice = index[RAD]

        mono = var[mono_index]
        cu1 = var[cu1_index]
        cu2 = var[cu2_index]
        dorm = var[dorm_slice]
        rad = var[rad_slice]

        kt2 = 2 * k_ter
        kp_mono = k_poly * mono
        ka_cu1 = k_act * cu1
        kd_cu2 = k_dorm * cu2
        sum_rad = np.sum(rad)
        kt2_rad = kt2 * rad

        num_var = len(var)
        jac = np.zeros((num_var, num_var))

        # monomer
        jac_mono = jac[mono_index]
        jac_mono[mono_index] = -k_poly * sum_rad
        jac_mono[rad_slice] = -kp_mono

        # dormant chains
        jac_dorm = jac[dorm_slice]
        np.fill_diagonal(jac_dorm[:, dorm_slice], -ka_cu1)
        np.fill_diagonal(jac_dorm[:, rad_slice], kd_cu2)

        # Cu(I)
        jac_cu1 = jac[cu1_index]
        jac_cu1[cu1_index] = -k_act * np.sum(dorm)
        jac_cu1[cu2_index] = k_dorm * sum_rad
        jac_cu1[dorm_slice] = -ka_cu1
        jac_cu1[rad_slice] = kd_cu2

        # Cu(II)
        jac[cu2_index] = -jac[cu1_index]

        # radicals
        jac_rad = jac[rad_slice]
        jac_rad[:, mono_index] = -k_poly * rad
        jac_rad[:, cu1_index] = k_act * dorm
        jac_rad[:, cu2_index] = -k_dorm * rad
        np.fill_diagonal(jac_rad[:, dorm_slice], ka_cu1)
        jac_rad_rad = jac_rad[:, rad_slice]
        np.fill_diagonal(jac_rad_rad, -(kp_mono + kd_cu2) - kt2_rad)
        np.fill_diagonal(jac_rad_rad[1:, :-1], kp_mono)
        jac_rad_rad -= kt2_rad[:, np.newaxis]

        # terminated chains
        if self.termination:
            # length 2 to n
            num_ter1 = max_rad_len - 1
            jac_ter1 = jac[index[TER_A], rad_slice]
            for p in range(num_ter1):
                p_slice = slice(None, p + 1)
                jac_ter1[p, p_slice] = rad[p_slice][::-1]
            for p in range(0, num_ter1, 2):
                jac_ter1[p, p // 2] *= 2
            jac_ter1 *= kt2

            # length n+1 to 2n
            num_ter2 = max_rad_len
            jac_ter2 = jac[index[TER_B], rad_slice]
            for p in range(num_ter2):
                p_slice = slice(p, None)
                jac_ter2[p, p_slice] = rad[p_slice][::-1]
            for p in range(0, num_ter2, 2):
                jac_ter2[p, p // 2] *= 2
            jac_ter2 *= kt2

        return jac

    def _generate_plot(self, key):
        values = self.quant[self.index[key]]
        len_values = len(values)
        if key == DORM:
            space = np.linspace(1, len_values, len_values)
            num = 1
            label = 'Dormant chains'
        elif key == RAD:
            space = np.linspace(1, len_values, len_values)
            num = 2
            label = 'Radical chains'
        elif key == TER:
            space = np.linspace(2, len_values + 1, len_values)
            num = 3
            label = 'Terminated chains'
        axis = plt.subplot(3, 1, num)
        plot = axis.plot(space, values, label=label)[0]
        axis.legend()
        self.axes[key] = axis
        self.plots[key] = plot

    def _update_plot(self, key):
        values = self.quant[self.index[key]]
        self.axes[key].set_ylim([0, np.max(values) * 1.1])
        self.plots[key].set_ydata(values)



import gym
from gym import spaces
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


__all__ = ['ATRPEnv']

''' rate constants '''
K_POLY  = 0
K_ACT   = 1
K_DEACT = 2
K_TER   = 3

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
STABLE = 'dorm_ter'

''' epsilon for float comparison '''
EPS = 1e-3

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
        k_poly:  rate constant for (monomer consumption);
        k_act:   rate constant for (dormant chain --> radical);
        k_deact: rate constant for (radical --> dormant chain);
        k_ter:   rate constant for (radical --> terminated chain).

    Observation related:
        observation_mode:
            'all':        capped indicators (of species to add), volume,
                          and quantities of all species;
            'all stable': capped indicators, volume, and quantities
                          of stable species.

    Action related:
        action_mode:
            'multi':  feeds multiple species in at each timestep;
            'single': feeds 1 species in at each timestep.
        mono_init:    initial quantity of monomer;
        mono_density: density of monomer (useful in calculating the volume);
        mono_unit:    unit amount of the "adding monomer" action;
        mono_cap:     maximum quantity (budget) of monomer.
        (Other X_init, X_unit, X_cap variables have similar definitions.)

    Reward related:
        reward_mode:
            'chain length' (cl): +1 reward once a unit amount of monomer
                                 is converted to chain lengths in this range;
            'distribution' (dn): rewards are based on distribution differences.
        reward_chain_type: type of chain that the reward is related with.
    'chain length' mode:
        cl_range: range of desired chain lengths
                  (left inclusive, right exclusive);
        cl_unit:  unit of change in equivalent amount of monomer
                  considered as rewarding.
    'distribution' mode:
        dn_dist:  desired distribution (of the rewarding chain type).

    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, timestep=1e1, completion_time=1e5,
                 max_completion_steps=10,
                 max_rad_len=100, termination=True,
                 k_poly=1e4, k_act=2e-2, k_deact=1e5, k_ter=1e10,
                 observation_mode='all', action_mode='single',
                 mono_init=9.0, mono_density=9.0, mono_unit=0.01, mono_cap=None,
                 cu1_init=0.2, cu1_unit=0.01, cu1_cap=None,
                 cu2_init=0.2, cu2_unit=0.01, cu2_cap=None,
                 dorm1_init=0.4, dorm1_unit=0.01, dorm1_cap=None,
                 sol_init=0.0, sol_density=1.0, sol_unit=0.01, sol_cap=0.0,
                 reward_mode='chain length', reward_chain_type='dorm',
                 cl_range=(20, 30), cl_unit=0.01,
                 dn_dist=None):
        rate_constant = {K_POLY: k_poly, K_ACT: k_act, K_DEACT: k_deact}
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

        # observation
        quant_len = 2 + 4 * max_rad_len if termination else 3 + 2 * max_rad_len
        max_chain_len = 2 * max_rad_len if self.termination else max_rad_len
        self.rad_chain_lengths = np.arange(1, 1 + max_rad_len)
        if termination:
            self.ter_chain_lengths = np.arange(2, 1 + 2 * max_rad_len)
        self.max_chain_len = max_chain_len
        observation_mode = observation_mode.lower()
        self.observation_mode = observation_mode
        if observation_mode == 'all':
            # 'capped' indicator of [MONO, CU1, CU2, DORM1, SOL],
            # volume and self.quant
            obs_len = 5 + 1 + quant_len
        if observation_mode == 'all stable':
            # 'capped' indicator of [MONO, CU1, CU2, DORM1, SOL],
            # volume, summed quantity of all stable chains, Cu(I), and Cu(II)
            obs_len = 5 + 1 + max_chain_len + 2
        self.observation_space = spaces.Box(0, np.inf, shape=(obs_len,))

        # build initial variable and timestep
        self.init_amount = {MONO: mono_init, CU1: cu1_init, CU2: cu2_init,
                            DORM1: dorm1_init, SOL: sol_init}
        self.density = {MONO: mono_density, SOL: sol_density}
        self.volume = volume = mono_init / mono_density + sol_init / sol_density
        quant_init = np.zeros(quant_len)
        quant_init[index[MONO]] = mono_init
        quant_init[index[CU1]] = cu1_init
        quant_init[index[CU2]] = cu2_init
        quant_init[index[DORM1]] = dorm1_init
        self.quant_init = quant_init
        self.step_time = np.array([0.0, timestep])
        self.completion_time = np.arange(0.0, completion_time + EPS, timestep)

        # actions
        action_mode = action_mode.lower()
        self.action_mode = action_mode
        if action_mode == 'multi':
            action_tuple = tuple(spaces.Discrete(2) for _ in range(5))
            action_space = spaces.Tuple(action_tuple)
        elif action_mode == 'single':
            action_space = spaces.Discrete(5)
        self.action_space = action_space
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
        if reward_chain_type == DORM or reward_chain_type == STABLE:
            chain_min = 1
        elif reward_chain_type == TER:
            chain_min = 2
        if self.reward_mode == 'chain length':
            reward_chain_type = reward_chain_type.lower()
            self.cl_unit = cl_unit
            start, end = cl_range[0] - chain_min, cl_range[1] - chain_min
            self.cl_slice = slice(start, end)
            self.cl_num_mono = np.arange(*cl_range)
        elif self.reward_mode == 'distribution':
            if dn_dist is None:
                chain_slice = index[reward_chain_type]
                dn_dist = np.ones(chain_slice.stop - chain_slice.start)
            dn_num_mono = np.arange(chain_min, chain_min + len(dn_dist))
            dn_mono_quant = dn_dist.dot(dn_num_mono)
            self.dn_target_quant = dn_dist / dn_mono_quant * mono_cap

        # rendering
        self.axes = None

    def _reset(self):
        self.added = self.init_amount.copy()
        self.quant = self.quant_init
        if self.reward_mode == 'chain length':
            chain = self.quant[self.index[self.reward_chain_type]]
            self.last_reward_chain = chain[self.cl_slice]
        return self.observation()

    def _step(self, action):
        if self.action_mode == 'single':
            action_list = [0] * self.action_space.n
            action_list[action] = 1
            action = tuple(action_list)
        self.take_action(action)
        done = self.done()
        info = {}
        if done:
            reward = self.run_atrp(self.completion_time)
        else:
            reward = self.run_atrp(self.step_time)
        observation = self.observation()
        #~ reward = self.reward(done)
        return observation, reward, done, info

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
            self.generate_plot(DORM)
            plt.title('Quantities')
            self.generate_plot(RAD)
            if self.termination:
                self.generate_plot(TER)
                stable_chains = self.stable_chains()
                self.generate_plot(STABLE, stable_chains)
            plt.xlabel('Chain length')
            plt.tight_layout()
        else:
            self.update_plot(DORM)
            self.update_plot(RAD)
            if self.termination:
                self.update_plot(TER)
                stable_chains = self.stable_chains()
                self.update_plot(STABLE, stable_chains)
        plt.draw()
        plt.pause(0.0001)

    def take_action(self, action):
        self.add(action, MONO, change_volume=True)
        self.add(action, CU1)
        self.add(action, CU2)
        self.add(action, DORM1)
        self.add_sol(action)

    def add(self, action, key, change_volume=False):
        if action[key] and not self.capped(key):
            unit = self.add_unit[key]
            self.quant[self.index[key]] += unit
            if change_volume:
                self.volume += self.volume_unit[key]
            self.added[key] += unit

    def add_sol(self, action):
        if action[SOL] and not self.capped(SOL):
            self.volume += self.volume_unit[SOL]
            self.added[SOL] += self.add_unit[SOL]

    def observation(self):
        capped = [self.capped(key) for key in [MONO, CU1, CU2, DORM1, SOL]]
        if self.observation_mode == 'all':
            obs = [capped, [self.volume], self.quant]
        elif self.observation_mode == 'all stable':
            stable_chains = self.stable_chains()
            quant = self.quant
            index = self.index
            cu1 = quant[index[CU1]]
            cu2 = quant[index[CU2]]
            obs = [capped, [self.volume], stable_chains, [cu1], [cu2]]
        return np.concatenate(obs)

    def stable_chains(self):
        quant = self.quant
        index = self.index
        max_rad_len = self.max_rad_len
        stable_chains = np.zeros(self.max_chain_len)
        stable_chains[:max_rad_len] = quant[index[DORM]]
        stable_chains[0] += quant[index[MONO]]
        if self.termination:
            stable_chains[1:] += quant[index[TER]]
        return stable_chains

    def done(self):
        return all(self.capped(key) for key in [MONO, CU1, CU2, DORM1, SOL])

    def reward(self):
        chain = self.chain(self.reward_chain_type)
        if self.reward_mode == 'chain length':
            reward_chain = chain[self.cl_slice]
            diff_reward_chain = reward_chain - self.last_reward_chain
            self.last_reward_chain = reward_chain
            return diff_reward_chain.dot(self.cl_num_mono)
        elif self.reward_mode == 'distribution':
            diff = chain - self.dn_target_quant
            return -diff.dot(diff)

    def chain(self, key):
        if key in [RAD, DORM, TER]:
            chain = self.quant[self.index[key]]
        elif key == STABLE:
            chain = self.stable_chains()
        return chain

    def capped(self, key):
        unit = self.add_unit[key]
        added_eps = self.added[key] + unit * EPS
        cap = self.add_cap[key]
        return cap is not None and added_eps > cap

    def run_atrp(self, step_time):
        # solve atrp odes to get new concentration
        volume = self.volume
        conc = self.quant / volume
        conc = odeint(self.atrp, conc, step_time, Dfun=self.atrp_jac)[1:]
        quant = conc * volume

        # adjust 'quant' so that the monomer amount is conserved
        index = self.index
        added = self.added
        ref_quant_eq_mono = added[MONO] + added[DORM1]
        reward = 0.0
        for qt in quant:
            mono = qt[index[MONO]]
            dorm = qt[index[DORM]]
            rad = qt[index[RAD]]
            quant_eq_mono = mono + (dorm + rad).dot(self.rad_chain_lengths)
            if self.termination:
                quant_eq_mono += qt[index[TER]].dot(self.ter_chain_lengths)
            qt *= ref_quant_eq_mono / quant_eq_mono
            self.quant = qt
            reward += self.reward()
        return reward

    def atrp(self, var, time):
        max_rad_len = self.max_rad_len

        rate_constant = self.rate_constant
        k_poly = rate_constant[K_POLY]
        k_act = rate_constant[K_ACT]
        k_deact = rate_constant[K_DEACT]
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
        dvar_dorm[:] = (k_deact * cu2) * rad - (k_act * cu1) * dorm

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

    def atrp_jac(self, var, time):
        max_rad_len = self.max_rad_len

        rate_constant = self.rate_constant
        k_poly = rate_constant[K_POLY]
        k_act = rate_constant[K_ACT]
        k_deact = rate_constant[K_DEACT]
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
        kd_cu2 = k_deact * cu2
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
        jac_cu1[cu2_index] = k_deact * sum_rad
        jac_cu1[dorm_slice] = -ka_cu1
        jac_cu1[rad_slice] = kd_cu2

        # Cu(II)
        jac[cu2_index] = -jac[cu1_index]

        # radicals
        jac_rad = jac[rad_slice]
        jac_rad[:, mono_index] = -k_poly * rad
        jac_rad[:, cu1_index] = k_act * dorm
        jac_rad[:, cu2_index] = -k_deact * rad
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

    def generate_plot(self, key, values=None):
        values = self.chain(key)
        len_values = len(values)
        if key == DORM:
            linspace = np.linspace(1, len_values, len_values)
            num = 1
            label = 'Dormant chains'
        elif key == RAD:
            linspace = np.linspace(1, len_values, len_values)
            num = 2
            label = 'Radical chains'
        elif key == TER:
            linspace = np.linspace(2, len_values + 1, len_values)
            num = 3
            label = 'Terminated chains'
        elif key == STABLE:
            linspace = np.linspace(1, len_values, len_values)
            num = 4
            label = 'All stable chains'
        num_plots = 4 if self.termination else 2
        axis = plt.subplot(num_plots, 1, num)
        plot = axis.plot(linspace, values, label=label)[0]
        if self.reward_mode == 'distribution':
            if key == self.reward_chain_type:
                target_quant = self.dn_target_quant
                target_label = 'Target distribution'
                axis.plot(linspace, target_quant, 'r', label=target_label)
        axis.legend()
        axis.set_xlim([0, self.max_chain_len])
        self.axes[key] = axis
        self.plots[key] = plot

    def update_plot(self, key, values=None):
        if values is None:
            values = self.quant[self.index[key]]
        self.axes[key].set_ylim([0, np.max(values) * 1.1])
        self.plots[key].set_ydata(values)



import gym, gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import ode


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

''' scale ymax in rendering by this factor '''
MARGIN_SCALE = 1.1

class ATRPBase(gym.Env):

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
        k_prop:  rate constant for (monomer consumption);
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

    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, max_rad_len=100, termination=True,
                 step_time=1e1, completion_time=1e5, min_steps=100,
                 k_prop=1e4, k_act=2e-2, k_deact=1e5, k_ter=1e10,
                 observation_mode='all',
                 mono_init=9.0, mono_density=9.0, mono_unit=0.01, mono_cap=None,
                 cu1_init=0.2, cu1_unit=0.01, cu1_cap=None,
                 cu2_init=0.2, cu2_unit=0.01, cu2_cap=None,
                 dorm1_init=0.4, dorm1_unit=0.01, dorm1_cap=None,
                 sol_init=0.0, sol_density=1.0, sol_unit=0.01, sol_cap=0.0,
                 **kwargs):
        # setup the simulation system
        # fundamental properties of the polymerization process
        self.max_rad_len = max_rad_len
        self.termination = termination

        # step related
        self.step_time = step_time
        self.completion_time = completion_time
        self.min_steps = min_steps

        # rate constants
        rate_constant = {K_POLY: k_prop, K_ACT: k_act, K_DEACT: k_deact}
        rate_constant[K_TER] = k_ter if termination else 0.0
        self.rate_constant = rate_constant

        # index (used in self.atrp and self.atrp_jac)
        self.index = self.init_index()

        # initial quant
        self.observation_mode = observation_mode.lower()
        self.init_amount = {MONO: mono_init, CU1: cu1_init, CU2: cu2_init,
                            DORM1: dorm1_init, SOL: sol_init}
        self.volume_init = mono_init / mono_density + sol_init / sol_density
        self.quant_init = self.init_quant()

        # actions
        self.action_pos = 0.0, 0.2, 0.4, 0.6, 0.8
        self.action_num = MONO, CU1, CU2, DORM1, SOL
        self.add_unit = {MONO: mono_unit, CU1: cu1_unit, CU2: cu2_unit,
                         DORM1: dorm1_unit, SOL: sol_unit}
        self.add_cap = {MONO: mono_cap, CU1: cu1_cap, CU2: cu2_cap,
                        DORM1: dorm1_cap, SOL: sol_cap}
        self.volume_unit = {MONO: mono_unit / mono_density,
                            SOL: sol_unit / sol_density}
        self._init_action(**kwargs)

        # initialize rewarding scheme (mostly in derived classes)
        self._init_reward(**kwargs)

        # rendering
        self.axes = None

        # ode integrator
        odeint = ode(self.atrp, self.atrp_jac)
        self.odeint = odeint.set_integrator('vode', method='bdf', nsteps=5000)

    def _reset(self):
        self.step_count = 0
        self.added = self.init_amount.copy()
        self.last_action = None
        self.quant = self.quant_init
        self.volume = self.volume_init
        return self.observation()

    def _step(self, action):
        self.step_count += 1
        action = self._parse_action(action)
        self.last_action = action
        self.take_action(action)
        done = self.done()
        info = {}
        run_time = self.completion_time if done else self.step_time
        self.run_atrp(run_time)
        reward = self._reward(done)
        observation = self.observation()
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
            num_plots = 5 if self.termination else 3
            self.init_plot(DORM, 1, num_plots)
            plt.title('Quantities')
            self.init_plot(RAD, 2, num_plots)
            if self.termination:
                self.init_plot(TER, 3, num_plots)
                stable_chains = self.stable_chains()
                self.init_plot(STABLE, 4, num_plots)
            action_axis = plt.subplot(num_plots, 1, num_plots)
            action_axis.get_xaxis().set_visible(False)
            action_axis.get_yaxis().set_visible(False)
            self.action_rect = {}
            action_labels = 'Monomer', 'Cu(I)', 'Cu(II)', 'Initiator', 'Solvent'
            zip_iter = zip(self.action_pos, action_labels, self.action_num)
            for pos, label, anum in zip_iter:
                color = 'y' if self.capped(anum) else 'r'
                rect = patches.Rectangle((pos, 0.0), 0.18, 1.0,
                                         color=color, fill=True)
                action_axis.add_patch(rect)
                action_axis.annotate(label, (pos + 0.03, 0.4))
                self.action_rect[pos] = rect
            plt.xlabel('Chain length')
            plt.tight_layout()
        else:
            self.update_plot(DORM)
            self.update_plot(RAD)
            if self.termination:
                self.update_plot(TER)
                self.update_plot(STABLE)
            last_action = self.last_action
            if last_action is not None:
                zip_iter = zip(self.action_pos, last_action, self.action_num)
                for pos, act, anum in zip_iter:
                    rect = self.action_rect[pos]
                    if self.capped(anum):
                        color = 'y'
                    else:
                        color = 'g' if act else 'r'
                    rect.set_color(color)
        plt.draw()
        plt.pause(0.0001)

    def init_index(self):
        max_rad_len = self.max_rad_len

        # variable indices (slices) for the ODE solver
        index = {MONO: 0, CU1: 1, CU2: 2, DORM1: 3}
        dorm_from = 3
        index[DORM] = slice(dorm_from, dorm_from + max_rad_len)
        rad_from = 3 + max_rad_len
        index[RAD] = slice(rad_from, rad_from + max_rad_len)

        if self.termination:
            # slices for the 2 types of terminated chains (ter)
            # ter type 1: length 2 to n
            ter1_from = 3 + 2 * max_rad_len
            index[TER_A] = slice(ter1_from, ter1_from + max_rad_len - 1)

            # ter type 2: length n+1 to 2n
            ter2_from = 2 + 3 * max_rad_len
            index[TER_B] = slice(ter2_from, ter2_from + max_rad_len)

            # total number of terminated chains is 2n-1
            index[TER] = slice(ter1_from, ter1_from + 2 * max_rad_len - 1)
        return index

    def init_quant(self):
        max_rad_len = self.max_rad_len
        # observation
        quant_len = 3 + 2 * max_rad_len
        max_chain_len = max_rad_len
        self.rad_chain_lengths = np.arange(1, 1 + max_rad_len)
        if self.termination:
            quant_len += 2 * max_rad_len - 1
            max_chain_len += max_rad_len
            self.ter_chain_lengths = np.arange(2, 1 + max_chain_len)
        self.max_chain_len = max_chain_len
        observation_mode = self.observation_mode
        if observation_mode == 'all':
            # 'capped' indicator of [MONO, CU1, CU2, DORM1, SOL],
            # volume and self.quant
            obs_len = 5 + 1 + quant_len
        elif observation_mode == 'all stable':
            # 'capped' indicator of [MONO, CU1, CU2, DORM1, SOL],
            # volume, summed quantity of all stable chains, Cu(I), and Cu(II)
            obs_len = 5 + 1 + max_chain_len + 2
        self.observation_space = gym.spaces.Box(0, np.inf, shape=(obs_len,))

        # build initial variable
        quant_init = np.zeros(quant_len)
        index = self.index
        init_amount = self.init_amount
        for key in MONO, CU1, CU2, DORM1:
            quant_init[index[key]] = init_amount[key]
        return quant_init

    def _init_action(self, *args, **kwargs):
        action_tuple = tuple(gym.spaces.Discrete(2) for _ in range(5))
        self.action_space = gym.spaces.Tuple(action_tuple)

    def _parse_action(self, action):
        return action

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
        capped = [self.capped(key) for key in (MONO, CU1, CU2, DORM1, SOL)]
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
        min_steps_exceeded = self.step_count >= self.min_steps
        all_capped = all(self.capped(s) for s in (MONO, CU1, CU2, DORM1, SOL))
        return min_steps_exceeded and all_capped

    def run_atrp(self, step_time):
        # solve atrp odes to get new concentration
        volume = self.volume
        conc = self.quant / volume
        odeint = self.odeint
        odeint.set_initial_value(conc, 0.0)
        conc = odeint.integrate(step_time)
        quant = conc * volume

        # adjust 'quant' so that the monomer amount is conserved
        index = self.index
        added = self.added
        ref_quant_eq_mono = added[MONO] + added[DORM1]
        mono = quant[index[MONO]]
        dorm = quant[index[DORM]]
        rad = quant[index[RAD]]
        quant_eq_mono = mono + (dorm + rad).dot(self.rad_chain_lengths)
        if self.termination:
            quant_eq_mono += quant[index[TER]].dot(self.ter_chain_lengths)
        ratio = ref_quant_eq_mono / quant_eq_mono if quant_eq_mono else 1.0
        quant *= ratio
        self.quant = quant

    def chain(self, key):
        quant = self.quant
        index = self.index
        if key in [RAD, DORM]:
            chain = quant[index[key]]
        elif key == TER:
            mono = quant[index[MONO]]
            ter = quant[index[TER]]
            chain = np.concatenate([[mono], ter])
        elif key == STABLE:
            chain = self.stable_chains()
        return chain

    def capped(self, key):
        unit = self.add_unit[key]
        added_eps = self.added[key] + unit * EPS
        cap = self.add_cap[key]
        return cap is not None and added_eps > cap

    def atrp(self, time, var):
        max_rad_len = self.max_rad_len

        rate_constant = self.rate_constant
        k_prop = rate_constant[K_POLY]
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
        kp_mono = k_prop * mono
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

    def atrp_jac(self, time, var):
        max_rad_len = self.max_rad_len

        rate_constant = self.rate_constant
        k_prop = rate_constant[K_POLY]
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
        kp_mono = k_prop * mono
        ka_cu1 = k_act * cu1
        kd_cu2 = k_deact * cu2
        sum_rad = np.sum(rad)
        kt2_rad = kt2 * rad

        num_var = len(var)
        jac = np.zeros((num_var, num_var))

        # monomer
        jac_mono = jac[mono_index]
        jac_mono[mono_index] = -k_prop * sum_rad
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
        jac_rad[:, mono_index] = -k_prop * rad
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

    def init_plot(self, key, num, num_plots):
        values = self.chain(key)
        len_values = len(values)
        chain_label_dict = {DORM: 'Dormant chains',
                            RAD: 'Radical chains',
                            TER: 'Terminated chains',
                            STABLE: 'All stable chains'}
        label = chain_label_dict[key]
        axis = plt.subplot(num_plots, 1, num)
        linspace = np.linspace(1, len_values, len_values)
        plot = axis.plot(linspace, values, label=label)[0]
        self._render_reward_init(key, axis)
        axis.legend()
        axis.set_xlim([0, self.max_chain_len])
        self.axes[key] = axis
        self.plots[key] = plot

    def update_plot(self, key):
        values = self.chain(key)
        ymax = np.max(values) * MARGIN_SCALE
        if not ymax:
            ymax = EPS
        axis = self.axes[key]
        axis.set_ylim([0, ymax])
        self._render_reward_update(key, axis)
        self.plots[key].set_ydata(values)

    def _init_reward(self, *args, **kwargs):
        pass

    def _reward(self, *args, **kwargs):
        return 0.0

    def _render_reward_init(self, *args, **kwargs):
        pass

    def _render_reward_update(self, *args, **kwargs):
        pass


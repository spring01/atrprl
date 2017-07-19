
import numpy as np
import gym.spaces
from .atrp_base import ATRPBase, MONO, DORM, MARGIN_SCALE


'''
ATRP simulation environment aiming at achieving a target distribution.
    Target is considered achieved if a Kolmogorov-Smirnov (KS) test
    on the ending distribution, assuming identical to the target distribution,
    cannot be rejected, in which case the environment gives a +1 reward.
KS test: determined whether a sample set comes from the same distribution
    KS statistic: D_n = sup_x |F_n(x) - F(x)|;
    KS test rejects null if \sqrt{n} * D_n > c(\alpha);
    c(\alpha) = 1.63 when \alpha = 0.01;
    In this env, n = `ks_num_sample`.

Actions:
    To simplify learning, actions are limited to "adding one species per action"
    Action space is Discrete(6).
    Mapping used by `_parse_action`:
        0 --> (0, 0, 0, 0, 0)
        1 --> (1, 0, 0, 0, 0)
        2 --> (0, 1, 0, 0, 0)
        3 --> (0, 0, 1, 0, 0)
        4 --> (0, 0, 0, 1, 0)
        5 --> (0, 0, 0, 0, 1)
Input arguments:
    reward_chain_type: type of chain that the reward is related with;
    dn_distribution:   target distribution (of the rewarding chain type);
    ks_num_sample:     number of sample used in KS test
'''

KS_FACTOR = 1.63 # corresponding value for alpha = 0.05

class ATRPTargetDistrib(ATRPBase):

    def _init_action(self, **kwargs):
        self.action_space = gym.spaces.Discrete(6)
        self.parse_action_dict = {0: (0, 0, 0, 0, 0),
                                  1: (1, 0, 0, 0, 0),
                                  2: (0, 1, 0, 0, 0),
                                  3: (0, 0, 1, 0, 0),
                                  4: (0, 0, 0, 1, 0),
                                  5: (0, 0, 0, 0, 1)}

    def _parse_action(self, action):
        return self.parse_action_dict[action]

    def _init_reward(self, reward_chain_type=DORM, dn_distribution=None,
                     ks_num_sample=2e3):
        reward_chain_type = reward_chain_type.lower()
        self.reward_chain_type = reward_chain_type
        self.dn_distribution = dn_distribution = np.array(dn_distribution)
        self.max_accept = KS_FACTOR / np.sqrt(ks_num_sample)
        dn_num_mono = np.arange(1, 1 + len(dn_distribution))
        dn_mono_quant = dn_distribution.dot(dn_num_mono)
        self.dn_num_mono = dn_num_mono
        mono_cap = self.add_cap[MONO]
        self.dn_target_quant = dn_distribution / dn_mono_quant * mono_cap
        self.target_ymax = np.max(self.dn_target_quant) * MARGIN_SCALE

    def _reward(self, done):
        if done:
            chain = self.chain(self.reward_chain_type)
            dn_distribution = self.dn_distribution
            target = dn_distribution / np.sum(dn_distribution)
            cdf_target = np.cumsum(target)
            current = chain / np.sum(chain)
            cdf_current = np.cumsum(current)
            ks_stat = np.max(np.abs(cdf_target - cdf_current))
            reward = float(ks_stat < self.max_accept)
        else:
            reward = 0.0
        return reward

    def _render_reward_init(self, key, axis):
        if key == self.reward_chain_type:
            target_quant = self.dn_target_quant
            target_label = 'Target distribution'
            len_values = len(target_quant)
            linspace = np.linspace(1, len_values, len_values)
            axis.plot(linspace, target_quant, 'r', label=target_label)

    def _render_reward_update(self, key, axis):
        if key == self.reward_chain_type:
            _, ymax = axis.get_ylim()
            if ymax < self.target_ymax:
                axis.set_ylim([0, self.target_ymax])


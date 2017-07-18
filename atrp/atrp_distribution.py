
import numpy as np
import gym.spaces
from .atrp_base import ATRPBase, MONO, MARGIN_SCALE


KS_NUM_SAMPLE = 5e3
KS_FACTOR = 1.36 # corresponding value for alpha = 0.05

'''
Action space is Discrete(6):
    0 --> (0, 0, 0, 0, 0)
    1 --> (1, 0, 0, 0, 0)
    2 --> (0, 1, 0, 0, 0)
    3 --> (0, 0, 1, 0, 0)
    4 --> (0, 0, 0, 1, 0)
    5 --> (0, 0, 0, 0, 1)
Reward based on difference between the ending/target distributions:
    reward_chain_type: type of chain that the reward is related with.
    dn_distribution:  desired distribution (of the rewarding chain type).
'''
class ATRPDistribution(ATRPBase):

    def _init_action(self, **kwargs):
        self.action_space = gym.spaces.Discrete(6)

    def _parse_action(self, action):
        parsed_action = [0] * 5
        if action:
            parsed_action[action - 1] = 1
        return parsed_action

    def _init_reward(self, reward_chain_type, dn_distribution):
        reward_chain_type = reward_chain_type.lower()
        self.reward_chain_type = reward_chain_type
        self.dn_distribution = dn_distribution = np.array(dn_distribution)
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
            ks_test = ks_stat < KS_FACTOR * np.sqrt(2.0 / KS_NUM_SAMPLE)
            print(ks_stat, KS_FACTOR * np.sqrt(2.0 / KS_NUM_SAMPLE))
            reward = float(ks_test)
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


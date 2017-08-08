
import numpy as np
import gym.spaces
from .atrp_base import ATRPBase, MONO, DORM, MARGIN_SCALE


'''
ATRP simulation environment aiming at achieving a target distribution.
    Target is considered achieved if the maximum difference in pdf is less than
    a given threshold. +1 reward if less than `thres_loose`, +2 if less than
    `thres_tight`.

Actions:
    To simplify learning, actions are limited to "adding one species per action"
    Action space is Discrete(6).
    Mapping used by `_parse_action`:
        0 --> (0, 0, 0, 0, 0) 'No-op'
        1 --> (1, 0, 0, 0, 0) 'Add a unit amount of monomer'
        2 --> (0, 1, 0, 0, 0) 'Add a unit amount of Cu(I)'
        3 --> (0, 0, 1, 0, 0) 'Add a unit amount of Cu(II)'
        4 --> (0, 0, 0, 1, 0) 'Add a unit amount of initiator (dorm1)'
        5 --> (0, 0, 0, 0, 1) 'Add a unit amount of solvent'
Input arguments:
    reward_chain_type: type of chain that the reward is related with;
    dn_distribution:   target distribution (of the rewarding chain type);
    thres_loose:       loose threshold for agreement of distributions;
    thres_tight:       tight threshold for agreement of distributions.
'''

class ATRPTargetDistribution(ATRPBase):

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
                     thres_loose=5e-3, thres_tight=2e-3,
                     reward_loose=0.1, reward_tight=1.0):
        reward_chain_type = reward_chain_type.lower()
        self.reward_chain_type = reward_chain_type
        self.dn_distribution = dn_distribution = np.array(dn_distribution)
        self.thres_loose = thres_loose
        self.thres_tight = thres_tight
        self.reward_loose = reward_loose
        self.reward_tight = reward_tight
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
            current = chain / np.sum(chain)
            max_diff = np.max(np.abs(target - current))
            reward = 0.0
            if max_diff < self.thres_loose:
                reward = self.reward_loose
            if max_diff < self.thres_tight:
                reward = self.reward_tight
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



import numpy as np
import pyemd
from scipy.spatial.distance import pdist, squareform
from .atrp_base import ATRPBase, MONO, MARGIN_SCALE


'''
Reward based on difference between the ending/target distributions:
    reward_chain_type: type of chain that the reward is related with.
    dn_distribution:  desired distribution (of the rewarding chain type).
    dn_distance_type: distance type (wrt. target); 'l2' or 'emd'
'''
class ATRPDistribution(ATRPBase):

    def init_reward(self, reward_chain_type, dn_distribution, dn_distance_type):
        reward_chain_type = reward_chain_type.lower()
        self.reward_chain_type = reward_chain_type
        dn_distribution = np.array(dn_distribution)
        dn_num_mono = np.arange(1, 1 + len(dn_distribution))
        dn_mono_quant = dn_distribution.dot(dn_num_mono)
        self.dn_num_mono = dn_num_mono
        mono_cap = self.add_cap[MONO]
        self.dn_target_quant = dn_distribution / dn_mono_quant * mono_cap
        self.target_ymax = np.max(self.dn_target_quant) * MARGIN_SCALE
        self.dn_distance_type = dn_distance_type.lower()

    def reward(self, done):
        if done:
            chain = self.chain(self.reward_chain_type)
            dn_target_quant = self.dn_target_quant
            if self.dn_distance_type == 'l2':
                diff = chain - dn_target_quant
                distance = diff.dot(diff)
            elif self.dn_distance_type == 'emd':
                dtn = chain / np.sum(chain)
                dtn_ref = dn_target_quant / np.sum(dn_target_quant)
                dist_mat = squareform(pdist(np.array([self.dn_num_mono]).T))
                distance = pyemd.emd(dtn, dtn_ref, dist_mat)
        else:
            reward = 0.0
        return reward

    def render_reward_init(self, key, axis):
        if key == self.reward_chain_type:
            target_quant = self.dn_target_quant
            target_label = 'Target distribution'
            len_values = len(target_quant)
            linspace = np.linspace(1, len_values, len_values)
            axis.plot(linspace, target_quant, 'r', label=target_label)

    def render_reward_update(self, key, axis):
        if key == self.reward_chain_type:
            _, ymax = axis.get_ylim()
            if ymax < self.target_ymax:
                axis.set_ylim([0, self.target_ymax])


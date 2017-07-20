
import numpy as np
import gym.spaces
from .atrp_base import ATRPBase, MONO, DORM, MARGIN_SCALE


'''
ATRP simulation environment aiming at maintaining peak position.
    Polymer products will be continuously released once the peak falls into
    the target range, and the process is terminated when the peak goes outside.

Actions:
    'No-op', 'Add monomer', 'Add initiator', 'Add both monomer and initiator'.
    At each timestep Cu(I) and Cu(II) are always added.
    Action space is Discrete(4).
    Mapping used by `_parse_action`:
        0 --> (0, 1, 1, 0, 0)
        1 --> (1, 1, 1, 0, 0)
        2 --> (0, 1, 1, 1, 0)
        3 --> (1, 1, 1, 1, 0)
Input arguments:
    reward_chain_type: type of chain that the reward is related with;
    mp_range:          target peak range (of the rewarding chain type).
'''

class ATRPMaintainPeak(ATRPBase):

    def _init_action(self, **kwargs):
        self.action_space = gym.spaces.Discrete(4)
        self.parse_action_dict = {0: (0, 1, 1, 0, 0),
                                  1: (1, 1, 1, 0, 0),
                                  2: (0, 1, 1, 1, 0),
                                  3: (1, 1, 1, 1, 0)}

    def _parse_action(self, action):
        return self.parse_action_dict[action]

    def _init_reward(self, reward_chain_type=DORM, mp_range=None):
        reward_chain_type = reward_chain_type.lower()
        self.reward_chain_type = reward_chain_type
        self.mp_range = mp_range

    def _reward(self, done):
        return 0.0

    def _render_reward_init(self, key, axis):
        if key == self.reward_chain_type:
            for pos in self.mp_range:
                axis.axvline(x=pos, color='r')


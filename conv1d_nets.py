
import numpy as np
import tensorflow.contrib.keras.api.keras.layers as kl
from tensorflow.contrib.keras.api.keras.initializers import RandomNormal
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.models import Model


'''
Input arguments:
    input_shape: Tuple of the format (dim_input,);
    num_actions: Number of actions in the environment; integer;
    net_arch:    Architecture of the actor-critic net.
'''
def conv_acnet(input_shape, num_actions, net_arch):
    state, feature = _conv_state_feature(input_shape)
    hidden = _feature_to_hidden(feature, net_arch)
    near_zeros = RandomNormal(stddev=1e-3)
    logits = kl.Dense(num_actions, kernel_initializer=near_zeros)(hidden)
    value = kl.Dense(1)(hidden)
    return Model(inputs=state, outputs=[value, logits])

'''
Input arguments:
    input_shape: Tuple of the format (dim_input,);
    num_actions: Number of actions in the environment; integer;
    net_arch:    Architecture of the q-net.
'''
def conv_qnet(input_shape, num_actions, net_arch):
    state, feature = _conv_state_feature(input_shape, net_arch)
    hidden = _feature_to_hidden(feature, net_arch)
    q_value = kl.Dense(num_actions)(hidden)
    return Model(inputs=state, outputs=q_value)

def _conv_state_feature(input_shape):
    dim_input, = input_shape
    input_shape = dim_input, 1
    state = kl.Input(shape=input_shape)
    Conv1D = kl.convolutional.Conv1D
    conv1 = Conv1D(8, 32, strides=2, activation='relu')(state)
    conv2 = Conv1D(8, 32, strides=1, activation='relu')(conv1)
    feature = kl.Flatten()(conv2)
    return state, feature

def _feature_to_hidden(feature, net_arch):
    hidden = feature
    for num_hid in net_arch:
        hidden = kl.Dense(num_hid, activation='relu')(hidden)
    return hidden

''' This interface, as it stands now, only works for stack size of 1. '''
def list_arrays_ravel_expand(state):
    return np.stack(state, axis=-1).ravel()[:, np.newaxis]



import numpy as np


'''
When a state is represented as a list of frames, this interface converts it
to a correctly shaped numpy array which can be fed into the neural network
'''
def list_frames_to_array(state):
    return np.stack(state, axis=-1).astype(np.float32)


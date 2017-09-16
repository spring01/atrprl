"""
ATRP console for human controlled ATRP reactor
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import importlib
import gym
import argparse
from hcdrl.common.envwrapper import HistoryStacker


episode_maxlen = 100000

def main():
    parser = argparse.ArgumentParser(description='ATRP console')

    # environment name
    parser.add_argument('--env', default='ATRP-ps-td-gv24-v0',
        help='Environment name')
    parser.add_argument('--env_import', default='atrp_ps_td_gv',
        help='File name where the environment is defined')
    parser.add_argument('--env_num_frames', default=1, type=int,
        help='Number of frames in a state')
    parser.add_argument('--env_act_steps', default=4, type=int,
        help='Do an action for how many steps')

    # parse arguments
    args = parser.parse_args()

    # environment
    importlib.import_module(args.env_import)
    env = gym.make(args.env)
    env = HistoryStacker(env, args.env_num_frames, args.env_act_steps)

    state = env.reset()
    for i in range(episode_maxlen):
        action = int(input())
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break



if __name__ == "__main__":
    main()

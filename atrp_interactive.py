"""
ATRP interactive console for human controlled ATRP reactor
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import importlib
import gym
import argparse
from hcdrl.common.envwrapper import HistoryStacker
import pygame
from pygame.locals import *
import pickle


episode_maxlen = 100000

def main():
    parser = argparse.ArgumentParser(description='ATRP interactive')

    # environment
    parser.add_argument('--env', default='ATRP-ps-v0',
        help='Environment name')
    parser.add_argument('--env_import', default='atrpenv.atrp_ps',
        help='File name where the environment is defined')
    parser.add_argument('--env_num_frames', default=1, type=int,
        help='Number of frames in a state')
    parser.add_argument('--env_act_steps', default=4, type=int,
        help='Do an action for how many steps')

    # state-action saving
    parser.add_argument('--save', default=None, type=str,
        help='Save state sequence and action sequence to file')

    # parse arguments
    args = parser.parse_args()

    # generate environment
    importlib.import_module(args.env_import)
    env = gym.make(args.env)
    env.unwrapped.action_parse = False
    env = HistoryStacker(env, args.env_num_frames, args.env_act_steps)

    # reset environment
    all_states = [env.reset()]
    env.render()
    all_actions = []
    all_rewards = []

    # detect key presses to step the environment
    noop_key = K_BACKQUOTE
    avail_key_list = [K_1, K_2, K_3, K_4, K_5]
    ini_pressed = {ak: False for ak in avail_key_list}
    down = ini_pressed.copy()
    pressed = ini_pressed.copy()
    for _ in range(episode_maxlen):
        pygame.time.wait(1)
        step = False
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit();
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == noop_key:
                action = tuple(0 for _ in avail_key_list)
                step = True
            elif event.key in down:
                down[event.key] = True
                pressed[event.key] = True
        if event.type == KEYUP and event.key in down:
            down[event.key] = False
            if not any(down[ak] for ak in avail_key_list):
                action = tuple(int(pressed[ak]) for ak in avail_key_list)
                step = True
        if step:
            down = ini_pressed.copy()
            pressed = ini_pressed.copy()
            all_actions.append(action)
            state, reward, done, info = env.step(action)
            all_rewards.append(reward)
            all_states.append(state)
            env.render()
            if done:
                break

    # save states and actions if requested
    if args.save is not None:
        with open(args.save, 'wb') as save:
            pickle.dump((all_states, all_actions, all_rewards), save)
        print('states and actions saved to {}'.format(args.save))

    # wait for user to close the pygame window
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit();
            break



if __name__ == '__main__':
    main()

"""
ATRP deep RL evaluator
Supports both DQN and A3C
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import importlib
import gym
import argparse
import numpy as np
import tensorflow as tf
from hcdrl.common.policy import EpsGreedy, Stochastic
from hcdrl.common.envwrapper import HistoryStacker
from hcdrl.common.interface import list_arrays_ravel
from hcdrl.common.neuralnet.qnet import QNet
from hcdrl.common.neuralnet.acnet import ACNet
from hcdrl.simple_nets import simple_acnet, simple_qnet
from conv1d_nets import conv_acnet, conv_qnet, list_arrays_ravel_expand


episode_maxlen = 100000

def main():
    parser = argparse.ArgumentParser(description='Deep RL ATRP')

    # environment name
    parser.add_argument('--env', default=None, required=True,
        help='Environment name')
    parser.add_argument('--env_import', default=None, required=True,
        help='File name where the environment is defined')
    parser.add_argument('--env_num_frames', default=1, type=int,
        help='Number of frames in a state')
    parser.add_argument('--env_act_steps', default=4, type=int,
        help='Do an action for how many steps')

    # policy arguments
    parser.add_argument('--policy_type', default='stochastic', type=str,
        choices=['epsilon greedy', 'stochastic'],
        help='Evaluation policy type')
    parser.add_argument('--policy_eps', default=0.0, type=float,
        help='Epsilon in epsilon-greedy policy')

    # neural net arguments
    parser.add_argument('--net_rl_type', default='acnet', type=str,
        choices=['qnet', 'acnet'],
        help='Neural net reinforcement learning algorithm type')
    parser.add_argument('--net_type', default='dense', type=str,
        choices=['dense', 'conv'],
        help='Neural net type')
    parser.add_argument('--net_arch', nargs='+', type=int, default=(100,),
        help='Neural net architecture')

    # trained weights
    parser.add_argument('--read_weights', default=None, type=str,
        help='Read weights from file')

    # evaluation
    parser.add_argument('--eval_episodes', default=20, type=int,
        help='Number of episodes in evaluation')

    # rendering
    parser.add_argument('--render', default='true', type=str,
        choices=['true', 't', 'false', 'f', 'end'],
        help='Do rendering or not')

    # print action sequence
    parser.add_argument('--print_action_seq', default='false', type=str,
        choices=['true', 't', 'false', 'f', 'end'],
        help='Whether the evaluator prints out the action sequence')

    # parse arguments
    args = parser.parse_args()
    render = args.render.lower()
    render_always = render in ['true', 't']
    render_end = render == 'end'

    # environment
    importlib.import_module(args.env_import)
    env = gym.make(args.env)
    env = HistoryStacker(env, args.env_num_frames, args.env_act_steps)
    input_shape = sum(sp.shape[0] for sp in env.observation_space.spaces),
    num_actions = env.action_space.n

    # neural net
    net_type = args.net_type.lower()
    if args.net_type == 'dense':
        acnet = simple_acnet
        qnet = simple_qnet
        interface = list_arrays_ravel
    elif args.net_type == 'conv':
        acnet = conv_acnet
        qnet = conv_qnet
        interface = list_arrays_ravel_expand
    net_rl_type = args.net_rl_type.lower()
    if net_rl_type == 'qnet':
        net_builder = lambda args: QNet(qnet(*args))
    elif net_rl_type == 'acnet':
        net_builder = lambda args: ACNet(acnet(*args))
    net_args = input_shape, num_actions, args.net_arch
    net = net_builder(net_args)
    sess = tf.Session()
    net.set_session(sess)
    sess.run(tf.global_variables_initializer())
    net.load_weights(args.read_weights)

    # policy
    if args.policy_type == 'epsilon greedy':
        policy = EpsGreedy(epsilon=args.policy_eps)
    elif args.policy_type == 'stochastic':
        policy = Stochastic()

    all_total_rewards = []
    for _ in range(args.eval_episodes):
        state = env.reset()
        if render_always:
            env.render()
        total_rewards = 0.0
        action_sequence = []
        for i in range(episode_maxlen):
            state = interface(state)
            action_values = net.action_values(np.stack([state]))[0]
            action = policy.select_action(action_values)
            action_sequence.append(action)
            state, reward, done, info = env.step(action)
            if render_always:
                env.render()
            total_rewards += reward
            if done:
                if render_end:
                    env.render()
                break
        all_total_rewards.append(total_rewards)
        print('episode reward:', total_rewards)
        if args.print_action_seq in ['true', 't']:
            print('Action sequence:')
            print(action_sequence)
    print('average episode reward:', np.mean(all_total_rewards))


if __name__ == "__main__":
    main()

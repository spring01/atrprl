"""
ATRP deep RL evaluator
Supports both DQN and A3C
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import gym
import argparse
import numpy as np
import tensorflow as tf
from hcdrl.common.policy import EpsGreedy, Stochastic
from hcdrl.common.neuralnet.qnet import QNet
from hcdrl.common.neuralnet.acnet import ACNet
from hcdrl.simple_nets import simple_acnet, simple_qnet
import atrp_ps


episode_maxlen = 100000

def main():
    parser = argparse.ArgumentParser(description='Deep RL ATRP')

    # environment name
    parser.add_argument('--env', default='ATRP-polystyrene-v0',
        help='Environment name')

    # policy arguments
    parser.add_argument('--policy_type', default='stochastic', type=str,
        choices=['epsilon greedy', 'stochastic'],
        help='Evaluation policy type')
    parser.add_argument('--policy_eps', default=0.0, type=float,
        help='Epsilon in epsilon-greedy policy')

    # neural net arguments
    parser.add_argument('--net_type', default='acnet', type=str,
        choices=['qnet', 'acnet'],
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
        choices=['true', 'True', 't', 'T', 'false', 'False', 'f', 'F'],
        help='Do rendering or not')

    # parse arguments
    args = parser.parse_args()
    render = args.render.lower() in ['true', 't']

    print('########## All arguments:', args)

    # environment
    env = gym.make(args.env)
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # neural net
    net_type = args.net_type.lower()
    if net_type == 'qnet':
        net_builder = lambda args: QNet(simple_qnet(*args))
    elif net_type == 'acnet':
        net_builder = lambda args: ACNet(simple_acnet(*args))
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
        if render:
            env.render()
        total_rewards = 0.0
        for i in range(episode_maxlen):
            action_values = net.action_values(np.stack([state]))[0]
            action = policy.select_action(action_values)
            state, reward, done, info = env.step(action)
            if render:
                env.render()
            total_rewards += reward
            if done:
                break
        all_total_rewards.append(total_rewards)
        print('episode reward:', total_rewards)
    print('average episode reward:', np.mean(all_total_rewards))


if __name__ == "__main__":
    main()

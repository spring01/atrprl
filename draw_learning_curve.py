
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)


all_episode_reward = []
max_step = 0
with open(sys.argv[1]) as out:
    for line in out:
        if 'episode reward' in line:
            reward = float(re.findall('\d+\.\d+', line)[0])
            all_episode_reward.append(reward)
        if 'training step' in line:
            max_step = int(re.findall('\d+', line)[0])
window_len = int(sys.argv[2]) if len(sys.argv) > 2 else 100
window = np.ones(window_len) / window_len
mean_all_episode_reward = np.convolve(all_episode_reward, window, mode='valid')
space = np.linspace(1, max_step, len(mean_all_episode_reward))
plt.plot(space, mean_all_episode_reward)
plt.xlabel('Number of simulator steps')
plt.ylabel('Average episodic reward')
if len(sys.argv) > 3:
    plt.savefig(sys.argv[3])
else:
    plt.show()


import sys
import re
import numpy as np
import matplotlib.pyplot as plt


all_episode_reward = []
with open(sys.argv[1]) as out:
    for line in out:
        if 'episode reward' in line:
            reward = float(re.findall('\d+\.\d+', line)[0])
            all_episode_reward.append(reward)
window_len = int(sys.argv[2]) if len(sys.argv) > 2 else 100
window = np.ones(window_len) / window_len
mean_all_episode_reward = np.convolve(all_episode_reward, window, mode='valid')
plt.plot(mean_all_episode_reward)
plt.show()

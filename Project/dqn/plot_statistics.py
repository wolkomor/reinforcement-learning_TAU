"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import argparse
import os
import platform
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name

if not platform.system() == 'Windows':
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt

import pickle


parser = argparse.ArgumentParser(description='DQN args')
parser.add_argument('--pkls', required=True, nargs='+', help='pickle files to load')
parser.add_argument('--labels', required=True, nargs='+', help='names for pickles for legend')

args = parser.parse_args()


plt.clf()
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward (past 100 episodes)')

for pkl, label in zip(args.pkls, args.labels):
    with open(pkl, 'rb') as f:
        saved_state = pickle.load(f)

    num_items = len(saved_state.stats["mean_episode_rewards"])

    mean_episodes = [item[1] for item in saved_state.stats["mean_episode_rewards"]]
    best_mean = [item[1] for item in saved_state.stats["best_mean_episode_rewards"]]
    t_mean = [item[0] for item in saved_state.stats["mean_episode_rewards"]]
    t_best = [item[0] for item in saved_state.stats["best_mean_episode_rewards"]]

    plt.plot(t_mean, mean_episodes, label='mean reward {}'.format(label))
    plt.plot(t_best, best_mean, label='best mean rewards {}'.format(label))
plt.legend()
plt.savefig('statistics.png')

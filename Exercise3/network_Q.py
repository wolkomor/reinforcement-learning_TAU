import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import deque

# Load environment
env = gym.make('FrozenLake-v0')


# Define Helper Functions
def eval(net, runs=100, render=False):
    total_reward = 0
    for i in range(runs):
        s = env.reset()
        if render: env.render()
        while True:
            a = net(one_hot(s)).max(0)[1].item()
            s, r, d, _ = env.step(a)
            if render: env.render()
            total_reward += r
            if d:
                break
    return total_reward / runs


def one_hot(i, size=env.observation_space.n):
    v = torch.zeros(size)
    v[i] = 1
    return v


# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
# TODO: define network, loss and optimiser(use learning rate of 0.1).
f = torch.nn.Sequential(nn.Linear(env.observation_space.n, env.action_space.n))

optimizer = torch.optim.Adam(f.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
loss_func = nn.MSELoss()

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
mini_batch_size = 50
eval_interval = num_episodes / 10
# create lists to contain total rewards and steps per episode
jList = []
rList = []

# Save experience for training
experience = deque(maxlen=1000)
# Initial Experience
while len(experience) < 1000:
    s = env.reset()
    for _ in range(100):
        a = env.action_space.sample()
        s1, r, done, _ = env.step(a)
        experience.append((s, a, r, s1, done))
        s = s1
        if done: break

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    if i % eval_interval == 0:
        before = eval(f)
    # The Q-Network
    # START Running an episode
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        # (run the network for current state and choose the action with the maxQ)
        # TODO: Implement Step 1
        Q = f(one_hot(s))
        # a = Q.argmax().item()
        a = Q.max(0)[1].item()

        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        
        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(a)
        experience.append((s, a, r, s1, d))
        
        rAll += r
        s = s1
        if d == True:
            # Reduce chance of random action as we train the model.
            e = 1. / ((i / 50) + 10)
            break
    jList.append(j)
    rList.append(rAll)
    # END Running an episode
    # START Train on mini-batch
    f.zero_grad()
    mini_batch_idx = np.random.choice(np.arange(len(experience)), size=mini_batch_size, replace=False)
    for id in mini_batch_idx:
        Q_val = f(one_hot(experience[id][0]))  # experience[i] = (s, a, r, s1, d)
        if experience[id][4]:  # if d
            Q_diff = experience[id][2] - Q_val[experience[id][1]]  # r - Q_val[a]
        else:
            # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
            # TODO: Implement Step 4
            Q1 = f(one_hot(experience[id][3]))  # f(s1)
            # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
            # TODO: Implement Step 5
            Q_tar = experience[id][2] + y * torch.max(Q1)  # r + y * maxQ1
            Q_diff = Q_tar.detach() - Q_val[experience[id][1]]  # target - Q_val[a]
        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        # TODO: Implement Step 6
        loss = torch.pow(Q_diff, 2) / (len(mini_batch_idx))
        loss.backward()
    optimizer.step()
    sched.step()
    # END Train on mini-batch
    # Evaluate progress
    if i % eval_interval == 0:
        after = eval(f)
        print(i, " : ", before, "->", after)

# Reports
print("Score over time (training): " + str(sum(rList) / len(rList)))
print("Learned Q Values:")
for i in range(16):
    print(f(one_hot(i)))

print("Average Reward after training:", eval(f, runs=1000))
env.close()

import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 50000
# create lists to contain total rewards and steps per episode
# jList = []
rList = []
# np.random.seed(123)


def choose_action(Q, s, epsilon=.1):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[s])


for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0  # Total reward during current episode
    d = False
    j = 0
    lr *= 0.9999
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # TODO: Implement Q-Learning
        # 1. Choose an action by greedily (with noise) picking from Q table
        e_t = 1 / (j ** 0.9)
        a = choose_action(Q, s, e_t)

        # 2. Get new state and reward from environment
        s_t1, r, d, info = env.step(a)
        # 3. Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s_t1]) - Q[s, a])
        s = s_t1
        # 4. Update total reward
        rAll += r
        # 5. Update episode if we reached the Goal State
        if d:
            break
    rList.append(rAll)


# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)


total_reward = 0
episodes = 10000
for i in range(episodes):
    s = env.reset()
    reward = 0
    while True:
        a = choose_action(Q, s, 0)
        s, r, done, _ = env.step(a)
        reward += r
        if done:
            break
    total_reward += reward


print("Percent of successful episodes after training:", total_reward * 100 / episodes)

env.close()

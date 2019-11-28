import gym
import matplotlib.pyplot as plt
import numpy as np


def agent(ot, w):
    """
    :param ot: 4-dimensional observation
    :param w: 4-dimensional vector of weight
    :return: 1, if ot ∗ w ≥ 0, otherwise 0
    """
    return 1 if np.dot(ot, w) >= 0 else 0


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    episodes_till_perfect_score = {}
    print("starting random search")

    # run random search 1000 times
    for i in range(1000):
        max_w = None
        max_reward = -np.inf
        print("random search run #{}".format(i))

        # random search
        for episode in range(10000):
            weights = np.random.uniform(-1, 1, (4,))
            observation = env.reset()
            agent_reward = 0

            # run an episode of the environment
            for step_t in range(200):
                env.render()
                action = agent(observation, weights)
                observation, reward, done, info = env.step(action)
                agent_reward += reward

                if done:
                    break

            if agent_reward > max_reward:
                max_w = weights
                max_reward = agent_reward
            if max_reward == 200:
                break
        episodes_till_perfect_score[i] = episode

    episodes_list = list(episodes_till_perfect_score.values())
    plt.hist(episodes_list, bins=list(range(200)))
    plt.xlabel("number of episodes required until score 200")
    plt.ylabel("count")
    plt.title("average: {}".format(np.mean(episodes_list)))
    plt.savefig("cart_pole.png")
    env.close()
    print('finished')

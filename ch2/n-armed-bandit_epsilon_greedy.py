import numpy as np
import matplotlib.pyplot as plt
'''
This code compares the performances of different
epsilons using epsilon-greedy strategy under stationary
situation.

In case of epsilon = 0, it works worst for both average reward and best hits count because it never explore, so 
it's unlikely for it to find the best action.

Epsilon = 0.1 outperforms epsilon = 0.01 in terms of average reward before around 1500 steps, but it is surpassed 
by the later one after it. Why it is surpassed? It is not so obvious, in fact, the best hits counts of epsilon_0.1 
is not surpassed by that of epsilon_0.01 util around 10000 steps. The reason is that although epsilon0.1 has more 
percentage of best action hits, the rest is not ideal as it is selected totally randomly. In contrast, epsilon0.01
is always 'second best' if it is not the best, and the it beats epsilon0.1 on average.
'''



class NArmedBandit(object):
    def __init__(self,  num_actions, epsilon, num_steps, with_noise = True):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.with_noise = with_noise
        self.reset()

    def reset(self):
        self.q = np.random.normal(0, 1, self.num_actions)
        self.avg_reward = np.zeros((self.num_actions,))
        self.cur_step = 1
        self.rewards = np.zeros(self.num_steps)
        self.occur = np.zeros((self.num_actions,))
        self.max_q = np.argmax(self.q)
        self.optimal_counts = np.zeros(self.num_steps)
        self.cur_optimal_counts = 0
        self.actions_order = np.zeros(self.num_steps)

    def find_action(self):
        p = np.random.uniform(0, 1)
        if p > self.epsilon:
            action = np.argmax(self.avg_reward)
        else:
            action = np.random.randint(0, 10)
        self.occur[action] += 1
        return action

    def get_reward(self, action):
        if self.with_noise is True:
            reward = self.q[action] + np.random.normal(0,1)
        else:
            reward = self.q[action]
        # new_estimate = old_estimation + 1/k * (target - old_estimate)
        self.avg_reward[action] = self.avg_reward[action] + 1/self.occur[action] * (reward - self.avg_reward[action])
        self.rewards[self.cur_step - 1] = reward
        if action == self.max_q:
            self.optimal_counts[self.cur_step - 1] += 1
        self.actions_order[self.cur_step - 1] = self.find_order(action)
        self.cur_step += 1
        return reward

    def find_order(self, action):
        sorted_indices = np.argsort(self.q)
        index = np.asscalar(np.where(sorted_indices == action)[0])
        return index

    def play(self):
        self.get_reward(self.find_action())

if __name__ == '__main__':

    num_times = 2000
    num_steps = 1000
    num_arms = 10
    epsilons = [0.1,0.01,0]
    num_epsilons = len(epsilons)
    with_noise = False
    bandits = [NArmedBandit(num_arms, eps, num_steps, with_noise) for eps in epsilons]
    avgs_reward_per_step = np.zeros((num_epsilons, num_steps))
    avgs_optimal_counts = np.zeros((num_epsilons, num_steps))
    avgs_actions_order = np.zeros((num_epsilons, num_steps))

    for t in range(num_times):
        for band in bandits:
            band.reset()
        for step in range(num_steps):
            for band in bandits:
                band.play()

        for i in range(num_epsilons):
            avgs_reward_per_step[i] = avgs_reward_per_step[i] + bandits[i].rewards
            avgs_optimal_counts[i] = avgs_optimal_counts[i] + bandits[i].optimal_counts
            avgs_actions_order[i] = avgs_actions_order[i] + bandits[i].actions_order

    avgs_reward_per_step = avgs_reward_per_step / num_times
    print(np.mean(avgs_reward_per_step))
    avgs_optimal_counts = avgs_optimal_counts / num_times
    avgs_actions_order = avgs_actions_order / num_times
    x = np.arange(num_steps)
    plt.subplot(3, 1, 1)
    for i in range(len(avgs_reward_per_step)):
        plt.plot(x, avgs_reward_per_step[i], label=epsilons[i])
    plt.legend()
    plt.subplot(3,1,2)
    for i in range(len(avgs_optimal_counts)):
        plt.plot(x, avgs_optimal_counts[i], label=epsilons[i])
    plt.legend()
    plt.subplot(3, 1, 3)
    for i in range(len(avgs_actions_order)):
        plt.plot(x, avgs_actions_order[i], label=epsilons[i])
    plt.legend()
    plt.show()
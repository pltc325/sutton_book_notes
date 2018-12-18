import numpy as np
import matplotlib.pyplot as plt
'''
This code compares sample average and 
constant alpha update rules using epsilon-greedy strategy
under a stationary situation.

We can see that sample average outperforms constant alpha, 
it's not a surprise as we have no reason to put emphasis on
recent actions as it's stationary.

Moreover, higher value of alphas will result in lower average rewards
as well as lower percentage of best action hits.
'''


class NArmedBandit(object):
    actions = [0,1,2]

    def __init__(self,  num_actions, epsilon, num_steps, alpha):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha
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

    def find_action(self):
        p = np.random.uniform(0, 1)
        if p > self.epsilon:
            action = np.argmax(self.avg_reward)
        else:
            action = np.random.randint(0, 10)
        self.occur[action] += 1
        return action

    def get_reward(self, action):
        reward = self.q[action] + np.random.normal(0,1)
        if self.alpha is None:
            self.avg_reward[action] = self.avg_reward[action] + 1/self.occur[action] * (reward - self.avg_reward[action])
        else:
            self.avg_reward[action] = self.avg_reward[action] + self.alpha * (reward - self.avg_reward[action])
        self.rewards[self.cur_step - 1] = reward
        if action == self.max_q:
            self.optimal_counts[self.cur_step - 1] += 1
        self.cur_step += 1
        return reward

    def play(self):
        self.get_reward(self.find_action())

if __name__ == '__main__':

    num_times = 2000
    num_steps = 1000
    num_arms = 10
    epsilon = 0.1
    alphas = [None, 0.1, 0.3, 0.6, 1]
    num_choices = len(alphas)
    bandits = [NArmedBandit(num_arms, epsilon, num_steps, alpha) for alpha in alphas]
    avgs_reward_per_step = np.zeros((num_choices, num_steps))
    avgs_optimal_counts = np.zeros((num_choices, num_steps))
    for t in range(num_times):
        for band in bandits:
            band.reset()
        for step in range(num_steps):
            for band in bandits:
                band.play()

        for i in range(num_choices):
            avgs_reward_per_step[i] = avgs_reward_per_step[i] + bandits[i].rewards
            avgs_optimal_counts[i] = avgs_optimal_counts[i] + bandits[i].optimal_counts

    avgs_reward_per_step = avgs_reward_per_step / num_times
    print(np.mean(avgs_reward_per_step))
    avgs_optimal_counts = avgs_optimal_counts / num_times
    x = np.arange(num_steps)
    plt.subplot(2, 1, 1)
    for i in range(len(avgs_reward_per_step)):
        plt.plot(x, avgs_reward_per_step[i], label=str(alphas[i]))
    plt.legend()
    plt.subplot(2,1,2)
    for i in range(len(avgs_optimal_counts)):
        plt.plot(x, avgs_optimal_counts[i], label=str(alphas[i]))
    plt.legend()
    plt.show()
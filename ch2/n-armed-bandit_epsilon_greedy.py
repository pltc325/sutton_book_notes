import numpy as np
import matplotlib.pyplot as plt
'''
This code compares the performances of different
epsilons using epsilon-greedy strategy under stationary
situation.

In case of epsilon = 0, it works worst for both average reward and best hits count because it never explore, so 
it's unlikely for it to find the best action.

Epsilon = 0.1 outperforms epsilon = 0.01 for all the experiments at first some thousands steps, but it is surpassed later.
This is because at early stage it spots good action very quickly since it explore more, but even if it finds the best action,
it will still choose non optimal action for the same reason. To the contrary, epsilon_0.01 find the best action slower, but
once it gets it, it sticks with it, hence gain more value ever after it.
'''


class NArmedBandit(object):
    def __init__(self, num_action, epsilon, num_step, with_noise=True):
        self.num_action = num_action
        self.epsilon = epsilon
        self.num_steps = num_step
        # whether or not the estimated action value will be added noise based on actual action value
        # with noise: Q = q + N(0,1) or without noise: Q = q
        self.with_noise = with_noise
        # actual action value
        self.q = None
        # estimated action value, approximated by average reward
        self.avg_reward = None
        # rewards obtained during steps
        self.rewards = None
        # number of each action's occurrences
        self.action_occur_count = None
        # max actual action value
        self.max_q = None
        # number of optimal action's occurrences
        self.optimal_action_occur_count = None
        # current step number
        self.cur_step = 1
        self.actions_order = None
        self.reset()

    def reset(self):
        self.q = np.random.normal(0, 1, self.num_action)
        self.avg_reward = np.zeros((self.num_action,))
        self.cur_step = 1
        self.rewards = np.zeros(self.num_steps)
        self.action_occur_count = np.zeros((self.num_action,))
        self.max_q = np.argmax(self.q)
        self.optimal_action_occur_count = np.zeros(self.num_steps)
        self.actions_order = np.zeros(self.num_steps)

    def choose_action(self):
        p = np.random.uniform(0, 1)
        if p > self.epsilon:
            action = np.argmax(self.avg_reward)
        else:
            action = np.random.randint(0, self.num_action)
        self.action_occur_count[action] += 1
        return action

    def get_reward(self, action):
        if self.with_noise is True:
            reward = self.q[action] + np.random.normal(0, 1)
        else:
            reward = self.q[action]
        # new_estimate =  (old_estimation * (k-1) + target) / k = old_estimation + 1/k * (target - old_estimate)
        self.avg_reward[action] = self.avg_reward[action] + 1 / self.action_occur_count[action] * (reward - self.avg_reward[action])
        self.rewards[self.cur_step - 1] = reward
        if action == self.max_q:
            self.optimal_action_occur_count[self.cur_step - 1] += 1
        self.actions_order[self.cur_step - 1] = self.find_order(action)
        self.cur_step += 1
        return reward

    def find_order(self, action):
        sorted_indices = np.argsort(self.q)
        index = np.asscalar(np.where(sorted_indices == action)[0])
        return index

    def play(self):
        self.get_reward(self.choose_action())


if __name__ == '__main__':
    num_times = 100
    num_steps = 4000
    num_arms = 10
    epsilons = [0.1, 0.01, 0]
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
            avgs_optimal_counts[i] = avgs_optimal_counts[i] + bandits[i].optimal_action_occur_count
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

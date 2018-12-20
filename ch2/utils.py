import matplotlib.pyplot as plt
import numpy as np


class NArmedBandit(object):
    def __init__(self, name, num_action, num_step, with_noise=True,
                 action_selection_strategy={'eps-greedy': {'eps': 0.1}}, Q_update_rule={'sample-average': None},
                 Q=None, is_stationary=True
                 ):
        self.name = name
        self.num_action = num_action
        self.num_step = num_step
        # whether or not the estimated action value will be added noise based on actual action value
        # with noise: Q = q + N(0,1) or without noise: Q = q
        self.with_noise = with_noise
        # actual action value
        self.q = None
        # estimated action value, approximated by average reward
        if Q is None:
            self.is_optimistic_init = False
        else:
            self.is_optimistic_init = True
        # rewards obtained during steps
        self.rewards = None
        # weight indicating how much the latest action has impact on the cumulative reward
        # number of each action's occurrences
        self.action_occur_count = None
        # max actual action value
        self.max_q = None
        # number of optimal action's occurrences
        self.optimal_action_occur_count = None
        # current step number
        self.cur_step = 1
        self.actions_order = None
        self.is_stationary = is_stationary
        self.is_updated = False
        self.parameter_Q = Q
        self.Q_update_rule = Q_update_rule
        self.action_selection_strategy = action_selection_strategy
        self.reset()

    def reset(self):
        """Reset before every trial

        Args:
            Q: initial estimates

        Returns: None

        """
        self.q = np.random.normal(0, 1, self.num_action)
        self.max_q = np.argmax(self.q)
        # print("**{1} max action q:{0}".format(np.argmax(self.q), self.name))
        if self.parameter_Q is None:
            self.Q = np.zeros((self.num_action,))
        else:
            # self.Q = np.ones((self.num_action,)) * 5
            self.Q = self.parameter_Q.copy()
        # print("Q:{0}".format(self.Q))
        self.cur_step = 1
        self.rewards = np.zeros(self.num_step)
        self.action_occur_count = np.zeros((self.num_action,))
        self.optimal_action_occur_count = np.zeros(self.num_step)
        self.actions_order = np.zeros(self.num_step)
        self.is_updated = False

    def update(self):
        """Update q.

        Update q to a new value, used as a way to simulate non-stationary environment.
        Only update once during one trial. It happens at the middle of the trial


        Returns: None

        """
        if not self.is_stationary:
            if self.cur_step > 0.5 * self.num_step:
                if self.is_updated is False:
                    self.q = np.random.normal(0, 1, self.num_action)
                    self.max_q = np.argmax(self.q)
                    self.is_updated = True

    def select_action(self):
        action = 0
        if 'eps-greedy' in self.action_selection_strategy:
            eps = self.action_selection_strategy['eps-greedy']['eps']
            p = np.random.uniform(0, 1)
            if p > eps:
                action = np.argmax(self.Q)
            else:
                action = np.random.randint(0, self.num_action)
        if 'UCB' in self.action_selection_strategy:
            action = 0
            found = False
            for i in range(len(self.action_occur_count)):
                if self.action_occur_count[i] == 0:
                    action = i
                    found = True
            if not found:
                c = self.action_selection_strategy['UCB']['c']
                Q_with_upper_bound = self.Q + c * np.sqrt(np.log(self.cur_step) / self.action_occur_count)
                action = np.argmax(Q_with_upper_bound)
        self.action_occur_count[action] += 1
        # print("{1} action chosen:{0}".format(action, self.name))
        return action

    def get_reward(self, action):
        if self.with_noise is True:
            reward = self.q[action] + np.random.normal(0, 1)
        else:
            reward = self.q[action]
        if 'sample-average' in self.Q_update_rule:
            # new_estimate =  (old_estimation * (k-1) + target) / k = old_estimation + 1/k * (target - old_estimate)
            self.Q[action] = self.Q[action] + 1 / self.action_occur_count[action] * (reward - self.Q[action])
        elif 'constant-alpha' in self.Q_update_rule:
            alpha = self.Q_update_rule['constant-alpha']
            self.Q[action] = self.Q[action] + alpha * (reward - self.Q[action])
        else:
            raise ValueError('Unrecognized Q update rule')

        self.rewards[self.cur_step - 1] = reward
        # print("{1} action chosen:{0} max_q:{2}".format(action, self.name, self.max_q))
        if action == self.max_q:
            self.optimal_action_occur_count[self.cur_step - 1] += 1
            # print("hit")

        self.actions_order[self.cur_step - 1] = self.find_order(action)
        self.cur_step += 1
        return reward

    def find_order(self, action):
        sorted_indices = np.argsort(self.q)
        index = np.asscalar(np.where(sorted_indices == action)[0])
        return index

    def play(self):
        self.get_reward(self.select_action())
        self.update()

    def parameter_info(self):
        return "action_selection_strategy:{0}/Q_update_rule:{1}/stationary:{2}/is_optimistic_init:{3}".format(self.action_selection_strategy,
                                                                                        self.Q_update_rule,
                                                                                        self.is_stationary,
                                                                                        self.is_optimistic_init)

    def show(self):
        print("alpha:{0}".format(self.alpha))
        print("q:{0}".format(self.q))
        print("Q:{0}".format(self.Q))
        print("q-Q:{0}".format(self.q - self.Q))


class TestBed(object):
    def __init__(self, bandits, num_time):
        self.bandits = bandits
        self.num_time = num_time
        self.sum_rewards = np.zeros((len(self.bandits), self.bandits[0].num_step))
        self.avg_rewards = np.zeros((len(self.bandits), self.bandits[0].num_step))
        self.sum_optimal_action_occur_count = np.zeros((len(self.bandits), self.bandits[0].num_step))
        self.avg_optimal_action_occur_count = np.zeros((len(self.bandits), self.bandits[0].num_step))

    def run(self):
        for t in range(self.num_time):
            for band in self.bandits:
                band.reset()
            # for step in range(band.num_step):
            #     for band in self.bandits:
            #         band.play()
            for band in self.bandits:
                for step in range(band.num_step):
                    band.play()

            for i in range(len(self.bandits)):
                self.sum_rewards[i] = self.sum_rewards[i] + self.bandits[i].rewards
                self.sum_optimal_action_occur_count[i] = self.sum_optimal_action_occur_count[i] + self.bandits[
                    i].optimal_action_occur_count

        self.avg_rewards = self.sum_rewards / self.num_time
        self.avg_optimal_action_occur_count = self.sum_optimal_action_occur_count / self.num_time

    def show(self):
        x = np.arange(self.bandits[0].num_step)
        plt.figure(figsize=(14, 9))
        plt.subplot(2, 1, 1)
        plt.title("Average reward")
        for i in range(len(self.avg_rewards)):
            plt.plot(x, self.avg_rewards[i], label=self.bandits[i].parameter_info())
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.title("Average optimal action percentage")
        for i in range(len(self.avg_optimal_action_occur_count)):
            plt.plot(x, self.avg_optimal_action_occur_count[i], label=self.bandits[i].parameter_info())
        plt.legend()

        plt.show()

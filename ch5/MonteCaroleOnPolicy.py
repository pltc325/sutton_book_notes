import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D

'''
MonteCarole On Policy
Very much similar to MonteCarole Exploring Starts except that when choosing action, we give 1 - epsilon + epsilon/|A(s)| 
probability to choose the greedy action, and epsilon/|A(s)| for the rest. 
'''

class BlackJackMonteCaroleAgent(object):
    def __init__(self,epsilon=0.1):
        self.epsilon = epsilon
        # ace-10
        dealer_diff_cards_num = 10
        # 12-21, cards under 12 don't count as player can always hit safely
        player_diff_cards_num = 10
        # the player either has usable ACE(used as 11) or not
        # note that this status can be changed during the game
        usable_num = 2
        action_num = 2
        self.usable_num = usable_num
        self.action_num = action_num
        self.dealer_diff_cards_num = dealer_diff_cards_num
        self.player_diff_cards_num = player_diff_cards_num
        self.stick = 0
        self.hit = 1
        self.q = np.zeros((action_num, usable_num, dealer_diff_cards_num, player_diff_cards_num))
        self.q_counts = np.zeros((action_num, usable_num, dealer_diff_cards_num, player_diff_cards_num))
        self.avg_q = np.zeros((action_num, usable_num, dealer_diff_cards_num, player_diff_cards_num))
        self.pi = np.zeros((usable_num, dealer_diff_cards_num, player_diff_cards_num))
        self.v = np.zeros((usable_num, dealer_diff_cards_num, player_diff_cards_num))
        #self.pi = np.ones((usable_num, dealer_diff_cards_num, player_diff_cards_num, action_num)) / action_num
        # mappings
        self.is_usable_to_index = {True:1, False:0}
        self.cur_num_to_index= {}
        for i in range(12,22):
            self.cur_num_to_index[i] = i - 12

    def take_action(self, ob):
        cur_sum, dealer_num, is_usable = ob
        # if under 12, hit
        if cur_sum < 12:
            return 1
        cur_sum_index = self.cur_num_to_index[cur_sum]
        is_usable_index = self.is_usable_to_index[is_usable]
        dealer_num_index = dealer_num - 1
        # otherwise follow policy
        p = np.random.rand()
        if p <= 1 - self.epsilon + self.epsilon / self.action_num:
            return self.pi[is_usable_index][dealer_num_index][cur_sum_index].astype(int)
        else:
            a = self.pi[is_usable_index][dealer_num_index][cur_sum_index].astype(int)
            print("a",a)
            candidates = list(range(self.action_num))
            print("candidates",candidates)
            candidates.remove(a)
            print("candidates",candidates)
            return random.choice(candidates)

    def update(self, obs_with_actions, rewards):
        rewards_actual = []
        max_reward = rewards[-1]
        rewards_actual.append(max_reward)
        for i in range(len(rewards)-1):
            rewards_actual.append(max_reward - rewards[i])
        # update q
        for i in range(len(obs_with_actions)):
            cur_sum, dealer_num, is_usable = obs_with_actions[i][0]
            action = obs_with_actions[i][1]
            if cur_sum < 12:
                continue
            cur_sum_index = self.cur_num_to_index[cur_sum]
            is_usable_index = self.is_usable_to_index[is_usable]
            dealer_num_index = dealer_num - 1
            self.q[action][is_usable_index][dealer_num_index][cur_sum_index] += rewards_actual[i]
            self.q_counts[action][is_usable_index][dealer_num_index][cur_sum_index] += 1
            self.avg_q[action][is_usable_index][dealer_num_index][cur_sum_index] = self.q[action][is_usable_index][dealer_num_index][cur_sum_index]/self.q_counts[action][is_usable_index][dealer_num_index][cur_sum_index]

        # update policy according to new q
        for ob_with_action in obs_with_actions:
            cur_sum, dealer_num, is_usable = ob_with_action[0]
            if cur_sum < 12:
                continue
            cur_sum_index = self.cur_num_to_index[cur_sum]
            is_usable_index = self.is_usable_to_index[is_usable]
            dealer_num_index = dealer_num - 1
            self.pi[is_usable_index][dealer_num_index][cur_sum_index] =  np.argmax(self.avg_q, axis=0)[is_usable_index][dealer_num_index][cur_sum_index]

    def compute_optimal_v(self):
        for i in range(self.usable_num):
            for j in range(self.dealer_diff_cards_num):
                for k in range(self.player_diff_cards_num):
                    self.v[i][j][k] = np.max(self.avg_q, axis=0)[i][j][k]


# helper function copied from OpenAI code, used for compute observation at the beginning of the game
def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


# same as above
def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


# plot function
# reproduce Figure 5.5, array element 1 means hit
# and array element 0 means stick
def plot_policy(values):
    pi_no_usable = values[0]
    pi_usable = values[1]
    array_4_print = np.zeros((10,10))
    # rotate 90 degree to left
    for i in range(10):
        array_4_print[:,i] = pi_no_usable[i][::-1]
    print(array_4_print)
    print()
    # rotate 90 degree to left
    for i in range(10):
        array_4_print[:,i] = pi_usable[i][::-1]
    print(array_4_print)


def plot_v(values):
    fig = plt.figure()
    x = np.arange(10)
    y = np.arange(10)
    x, y = np.meshgrid(x, y)
    is_usable_titles = {0:'Do Not Have Usable ACE' , 1:'Do Have Usable ACE'}
    for k in range(2):
        ax = fig.add_subplot(2, 1, k+1, projection='3d')
        ax.set_title(is_usable_titles[k])
        ax.plot_surface(x, y, values[k])
    plt.show()

if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    agent = BlackJackMonteCaroleAgent(0.1)
    episode_nums = [2000000]
    avgs_state_value_functions = []
    for episode_num in episode_nums:
        for i in range(episode_num):
            print("cur episode:", i)
            reward_sum = 0.0
            env.reset()
            # see what the value of the two cards the player already has at beginning
            cur_sum = sum_hand(env.player)
            # construct the first observation
            ob = cur_sum, env.dealer[0], usable_ace(env.player)
            done = False
            this_suite_obs_with_action = []
            this_suite_rewards = []
            while not done:
                action = agent.take_action(ob)
                print("action", action)
                old_ob_with_action = ob, action
                this_suite_obs_with_action.append(old_ob_with_action)
                ob, reward, done, _ = env.step(action)
                reward_sum += reward
                this_suite_rewards.append(reward_sum)
            # update state value function statistic using ob right before done
            agent.update(this_suite_obs_with_action, this_suite_rewards)
    agent.compute_optimal_v()
    print(np.sum(agent.v[0]))
    plot_policy(agent.pi)
    plot_v(agent.v)

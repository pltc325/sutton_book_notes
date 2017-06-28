import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
This code illustrates when given a monte carole policy, how we
evaluate it.

The four figures corresponds to Figure 5.2 in the book:
http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf.

The simulator used is the one provided by OpenAI, 
https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

The policy used is that the player sticks when 20 ro 21 and hits otherwise.
'''


class BlackJackMonteCaroleAgent(object):
    def __init__(self):
        # ace-10
        dealer_diff_cards_num = 10
        # 12-21, cards under 12 don't count as player can always hit safely
        player_diff_cards_num = 10
        # the player either has usable ACE(used as 11) or not
        # note that this status can be changed during the game
        usable_num = 2
        self.stick = 0
        self.hit = 1
        self.state_value_functions = np.zeros((usable_num, dealer_diff_cards_num, player_diff_cards_num))
        self.state_value_counts = np.zeros((usable_num, dealer_diff_cards_num, player_diff_cards_num))
        # self.avg_state_value_functions = self.state_value_functions/self.state_value_counts,
        # this is what we want to evaluate
        self.avg_state_value_functions = np.zeros((usable_num, dealer_diff_cards_num, player_diff_cards_num))
        # mappings
        self.is_usable_to_index = {True:1, False:0}
        self.cur_num_to_index= {}
        for i in range(12,22):
            self.cur_num_to_index[i] = i - 12

    def take_action(self, ob):
        cur_sum, dealer_num, is_usable = ob
        if cur_sum == 20 or cur_sum == 21:
            action = self.stick
        else:
            action = self.hit
        return action

    def update(self, obs, rewards):
        rewards_actual = []
        max_reward = rewards[-1]
        rewards_actual.append(max_reward)
        for i in range(len(rewards)-1):
            rewards_actual.append(max_reward - rewards[i])

        for i in range (len(obs)):
            cur_sum, dealer_num, is_usable = obs[i]
            if cur_sum < 12:
                continue
            cur_sum_index = self.cur_num_to_index[cur_sum]
            is_usable_index = self.is_usable_to_index[is_usable]
            dealer_num_index = dealer_num - 1
            self.state_value_counts[is_usable_index][dealer_num_index][cur_sum_index] += 1
            self.state_value_functions[is_usable_index][dealer_num_index][cur_sum_index] += rewards_actual[i]

    def summarize(self):
        self.avg_state_value_functions = self.state_value_functions / self.state_value_counts
        return self.avg_state_value_functions


# helper function copied from OpenAI code, used for compute observation at the beginning of the game
def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


# same as above
def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


# helper function used to plot
def plot(values):
    x = np.arange(10)
    y = np.arange(10)
    x, y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(8, 8))
    n = 1
    episode_num_titles = {0:"10000 episodes", 1:"500000 episodes"}
    is_usable_titles = {0:'Do Not Have Usable ACE' , 1:'Do Have Usable ACE'}
    for i in range(len(values)):
        for k in range(2):
            ax = fig.add_subplot(2, 2, n, projection='3d')
            ax.set_title(is_usable_titles[k] + ', ' + episode_num_titles[i])
            ax.plot_surface(x, y, values[i][k])
            n += 1
    plt.show()

if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    agent = BlackJackMonteCaroleAgent()
    episode_nums = [10000, 500000]
    avgs_state_value_functions = []
    for episode_num in episode_nums:
        for i in range(episode_num):
            reward_sum = 0.0
            env.reset()
            # see what the value of the two cards the player already has at beginning
            cur_sum = sum_hand(env.player)
            # construct the first observation
            ob = cur_sum, env.dealer[0], usable_ace(env.player)
            done = False
            this_suite_obs = []
            this_suite_rewards = []
            while not done:
                action = agent.take_action(ob)
                old_ob = ob
                this_suite_obs.append(old_ob)
                ob, reward, done, _ = env.step(action)
                reward_sum += reward
                this_suite_rewards.append(reward_sum)
            # update state value function statistic using ob right before done
            agent.update(this_suite_obs, this_suite_rewards)

        avgs_state_value_functions.append(agent.summarize())
    plot(avgs_state_value_functions)
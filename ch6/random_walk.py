import random

import matplotlib.pyplot as plt
import numpy as np

'''
Try to reproduce Figure 6.2 left and right
'''
class RandomWalkAgent(object):
    def __init__(self,alpha=0.01, episode_num=100, method="TD"):
        self.episode_num = episode_num
        self.alpha = alpha
        self.method = method
        self.left = 0
        self.right = 1
        self.actions = [-1,1]
        self.states_num = 7
        self.terminal_states = [0, 6]
        self.state_value_functions = np.ones(self.states_num)/2
        self.state_value_functions[0] = 0
        self.state_value_functions[6] = 0
        self.reset()

    def reset(self):
        self.cur_state = 3

    def take_action(self, ob):
        return random.choice(self.actions)

    def update_mc(self, obs, rewards):
        rewards_actual = []
        max_reward = rewards[-1]
        rewards_actual.append(max_reward)
        for i in range(len(rewards)-1):
            rewards_actual.append(max_reward - rewards[i])
        seen = []
        for i in range (len(obs)):
            s = obs[i]
            if s in seen:
                continue
            else:
                seen.append(s)
                self.state_value_functions[s] = self.state_value_functions[s] + self.alpha*(rewards_actual[i] - self.state_value_functions[s])

    def update_td(self, old_ob, new_ob, reward):
        self.state_value_functions[old_ob] += self.alpha*(reward + self.state_value_functions[new_ob] - self.state_value_functions[old_ob])

    def step(self,action):
        done = False
        self.cur_state = self.cur_state + action
        reward = 0
        if self.cur_state in self.terminal_states:
            done = True
            if self.cur_state == 6:
                reward = 1
        return self.cur_state, reward, done

    def walk(self):
        for i in range(self.episode_num):
            reward_sum = 0.0
            self.reset()
            done = False
            ob = self.cur_state
            this_suite_obs = []
            this_suite_rewards = []
            while not done:
                action = self.take_action(ob)
                old_ob = ob
                this_suite_obs.append(old_ob)
                ob, reward, done = self.step(action)
                reward_sum += reward
                this_suite_rewards.append(reward_sum)
                if self.method == "TD":
                    self.update_td(old_ob, ob, reward)
            # update state value function statistic using ob right before done
            if self.method == "MC":
                agent.update_mc(this_suite_obs, this_suite_rewards)

def rms(a,b):
    return np.sqrt(np.mean((b - a) ** 2))

if __name__ == "__main__":
    # figure 6.1 left
    state_value_functions = []
    # true value
    true_value = [0,1.0/6,2.0/6,3.0/6,4.0/6,5.0/6,0]
    state_value_functions.append(true_value)
    agent = RandomWalkAgent(alpha=0.1, episode_num=100, method="TD")
    episode_nums = [0, 1, 10, 100]
    for episode_num in episode_nums:
        agent = RandomWalkAgent(alpha=0.1, episode_num=episode_num, method="TD")
        agent.walk()
        print(agent.state_value_functions)
        state_value_functions.append(agent.state_value_functions)
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.plot(state_value_functions[0][1:6], label="true value")
    for i in range(len(episode_nums)):
        label = "episode num=" + str(episode_nums[i])
        # first one is true value, already plotted above, so start from i+1
        # leftmost, rightmost points not plotted as they're terminal states, so [1:6]
        plt.plot(state_value_functions[1+i][1:6],label=label)
    plt.legend()

    # figure 6.2 right
    seq_num = 100
    alphas_mc = [0.01, 0.02, 0.03, 0.04]
    alphas_td = [0.15, 0.1, 0.05]

    episode_num_max = 100
    data_mc = np.zeros((len(alphas_mc), episode_num_max, seq_num))
    data_td = np.zeros((len(alphas_mc), episode_num_max, seq_num))

    for i in range(len(alphas_mc)):
        for episode_num in range(episode_num_max):
            for j in range(seq_num):
                print(i,episode_num,j)
                agent = RandomWalkAgent(alpha=alphas_mc[i], episode_num=episode_num, method="MC")
                agent.walk()
                r = rms(agent.state_value_functions,true_value)
                data_mc[i][episode_num][j] = r
    for i in range(len(alphas_td)):
        for episode_num in range(episode_num_max):
            for j in range(seq_num):
                print(i,episode_num,j)
                agent = RandomWalkAgent(alpha=alphas_td[i], episode_num=episode_num, method="TD")
                agent.walk()
                r = rms(agent.state_value_functions,true_value)
                data_td[i][episode_num][j] = r
    plt.subplot(1, 2, 2)
    for i in range(len(alphas_mc)):
        plt.plot(np.mean(data_mc[i], axis=1), label="alpha=" + str(alphas_mc[i]))
    for i in range(len(alphas_td)):
        plt.plot(np.mean(data_td[i], axis=1), linestyle="--", label="alpha=" + str(alphas_td[i]))
    plt.legend()
    plt.show()
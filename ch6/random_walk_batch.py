import random

import matplotlib.pyplot as plt
import numpy as np

'''
Try to reproduce Figure 6.3
'''


class RandomWalkAgent(object):
    def __init__(self,alpha = 0.01,episode_num=100, method="TD"):
        self.episode_num = episode_num
        self.alpha = alpha
        self.method = method
        self.left = 0
        self.right = 1
        self.actions = [-1,1]
        self.states_num = 7
        self.terminal_states = [0, 6]
        self.reset()
        self.state_value_functions_list = np.zeros((episode_num,self.states_num))
        self.cur_state = 3
        self.state_value_functions = np.ones(self.states_num) / 2

    def reset(self):
        self.cur_state = 3
        self.state_value_functions = np.ones(self.states_num) / 2
        self.state_value_functions[0] = 0
        self.state_value_functions[6] = 0

    def take_action(self, ob):
        return random.choice(self.actions)

    def update_mc(self, suite_obs_rewards_so_far):
        # update according to current batch
        # current batch contains episodes seen so far, each episode := suite_obs_rewards_so_far[i]
        batch_used_times = 100
        for t in range(batch_used_times):
            increments = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for i in range(len(suite_obs_rewards_so_far)):
                obs, rewards = suite_obs_rewards_so_far[i]
                rewards_actual = []
                max_reward = rewards[-1]
                rewards_actual.append(max_reward)
                for k in range(len(rewards)-1):
                    rewards_actual.append(max_reward - rewards[k])
                seen = []
                for k in range (len(obs)):
                    s = obs[k]
                    if s in seen:
                        continue
                    else:
                        seen.append(s)
                        increments[s] += self.alpha * (rewards_actual[k] - self.state_value_functions[s])
            # update occurs when a batch being finished
            for s in range(1, 6):
                self.state_value_functions[s] += increments[s]

        self.state_value_functions_list[len(suite_obs_rewards_so_far)-1]=self.state_value_functions.copy()

    def update_td(self, suite_obs_rewards_so_far):
        batch_used_times = 100
        for t in range(batch_used_times):
            increments = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for i in range(len(suite_obs_rewards_so_far)):
                old_obs_obs, rewards = suite_obs_rewards_so_far[i]
                for i in range(len(old_obs_obs)):
                    old_ob, new_ob = old_obs_obs[i]
                    reward = rewards[i]
                    increments[old_ob] += self.alpha * (reward + self.state_value_functions[new_ob] - self.state_value_functions[old_ob])

            for s in range(1, 6):
                self.state_value_functions[s] += increments[s]
        self.state_value_functions_list[len(suite_obs_rewards_so_far)-1]=self.state_value_functions.copy()

    def step(self, action):
        done = False
        self.cur_state = self.cur_state + action
        reward = 0
        if self.cur_state in self.terminal_states:
            done = True
            if self.cur_state == 6:
                reward = 1
        return self.cur_state, reward, done

    def walk(self):
        suite_obs_reward_along_episodes_mc = []
        suite_obs_reward_along_episodes_td = []
        for i in range(self.episode_num):
            reward_sum = 0.0
            self.reset()
            done = False
            ob = self.cur_state
            this_suite_obs_mc = []
            this_suite_obs_td = []
            this_suite_reward_sums_mc = []
            this_suite_rewards_td = []
            while not done:
                action = self.take_action(ob)
                old_ob = ob
                this_suite_obs_mc.append(old_ob)
                ob, reward, done = self.step(action)
                this_suite_obs_td.append((old_ob,ob))
                reward_sum += reward
                this_suite_reward_sums_mc.append(reward_sum)
                this_suite_rewards_td.append(reward)
            suite_obs_reward_along_episodes_mc.append((this_suite_obs_mc,this_suite_reward_sums_mc))
            suite_obs_reward_along_episodes_td.append((this_suite_obs_td,this_suite_rewards_td))
            if self.method == "MC":
                agent.update_mc(suite_obs_reward_along_episodes_mc)
            if self.method == "TD":
                agent.update_td(suite_obs_reward_along_episodes_td)


def rms(aa, b):
    r = np.zeros(len(aa))
    for i in range(len(aa)):
        r[i] = np.sqrt(np.mean((b - aa[i]) ** 2))
    return r

if __name__ == "__main__":
    seq_num = 20
    alphas_mc = 0.001
    alphas_td = 0.001
    episode_num_max = 100
    data_mc = np.zeros((seq_num,episode_num_max))
    data_td = np.zeros((seq_num, episode_num_max))
    true_value = [0, 1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6, 0]

    print("MC")
    for j in range(seq_num):
        print(j)
        agent = RandomWalkAgent(alpha=alphas_mc, episode_num=episode_num_max, method="MC")
        agent.walk()
        rs = rms(agent.state_value_functions_list, true_value)
        data_mc[j] = rs

    print("TD")
    for j in range(seq_num):
        print(j)
        agent = RandomWalkAgent(alpha=alphas_td, episode_num=episode_num_max, method="TD")
        agent.walk()
        rs = rms(agent.state_value_functions_list, true_value)
        data_td[j] = rs

    plt.plot(np.mean(data_mc, axis=0), label="MC/" + str(alphas_mc))
    plt.plot(np.mean(data_td, axis=0), label="TD/" + str(alphas_td))

    plt.legend()
    plt.show()
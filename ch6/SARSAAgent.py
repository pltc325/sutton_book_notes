import numpy as np

from ch6.Agent import Agent


class SARSAAgent(Agent):
    def __init__(self, state_num, action_num, start_states, terminal_states, choose_action_strategy, epsilon, alpha, gamma, episode_num, row_num, col_num, step_max):
        super().__init__(state_num, action_num, start_states, terminal_states, choose_action_strategy, epsilon, alpha, gamma, episode_num, row_num, col_num, step_max)

    def update(self, old_ob, new_ob, action, reward):
        new_action = self.choose_action(new_ob)
        #print("old_ob",old_ob,"action",action,"new_ob",new_ob,"new_action",new_action)
        self.qsa[old_ob][action] = self.qsa[old_ob][action] + self.alpha*(reward  + self.gamma*self.qsa[new_ob][new_action]-self.qsa[old_ob][action])
        self.pi[old_ob] = np.argmax(self.qsa[old_ob])
        return new_action

    def run(self):
        for i in range(self.episode_num):
            self.reset()
            done = False
            ob = self.cur_state
            action = self.choose_action(ob)
            step = 0
            while not done:
                new_ob, reward, done = self.step(action)
                #new_action = self.update(ob, new_ob, action, reward)
                self.update(ob, new_ob, action, reward)
                ob = new_ob
                action = self.choose_action(new_ob)
                step += 1
                if step > self.step_max:
                    break
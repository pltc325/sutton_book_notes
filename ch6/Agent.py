import numpy as np

from ch6.Utils import Utils

'''
Try to reproduce Figure 6.4 using Sarsa: On-Policy TD Control
'''
class Agent(object):
    def __init__(self, state_num, action_num, start_states, terminal_states, choose_action_strategy, epsilon, alpha, gamma, episode_num, row_num, col_num, step_max):
        self.state_num = state_num
        self.action_num = action_num
        self.start_states = start_states
        self.terminal_states = terminal_states
        self.choose_action_strategy = choose_action_strategy
        self.qsa = np.zeros((state_num, action_num))
        self.pi = np.zeros(state_num)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode_num = episode_num
        self.row_num = row_num
        self.col_num = col_num
        self.step_max = step_max
        self.reset()
        self.cur_state = 0
        pass

    def reset(self):
        pass

    def choose_action(self, ob):
        if self.choose_action_strategy == 'epsilon_greedy':
            p = np.random.rand()
            if p <= 1 - self.epsilon:
                return int(self.pi[int(ob)])
            else:
                return int(np.random.randint(0, self.action_num))
        else:
            pass

    def update(self, old_ob, new_ob, action, reward):
        pass

    def step(self, action):
        pass

    def run(self):
        pass

    def run_greedily(self):
        self.reset()
        done = False
        ob = self.cur_state
        step = 0
        while not done:
            action = self.pi[ob]
            print(Utils.state2rowcol(ob, self.row_num, self.col_num),action )
            new_ob, reward, done = self.step(action)
            ob = new_ob
            step += 1
            if step > self.step_max:
                raise Exception("current policy may not be a good one as it runs too many steps yet doesn't finish.")
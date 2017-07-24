import numpy as np

from ch6.sarsa_agent import SARSAAgent
from ch6.utils1 import Utils


class WindyAgent(SARSAAgent):
    def __init__(self, state_num, action_num, start_states, terminal_states, choose_action_strategy, epsilon, alpha, gamma, episode_num, row_num, col_num, step_max):
        super().__init__(state_num, action_num, start_states, terminal_states,choose_action_strategy, epsilon, alpha,gamma, episode_num, row_num, col_num, step_max)
        self.wind_power = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

    def reset(self):
        self.cur_state = np.random.choice(self.start_states, 1)[0]

    '''
    0:up, 1:left, 2:down, 3:right
    '''
    def step(self, action):
        row, col = Utils.state2rowcol(self.cur_state, self.row_num,
                                      self.col_num)
        if action == 1:  # left
            row = max(0, row - self.wind_power[col])
            col = max(0, col - 1)
        if action == 0:  # up
            row = max(0, row - (1 + self.wind_power[col]))
        if action == 2:  # down
            if 1 - self.wind_power[col] >= 0:
                row = min(self.row_num - 1, row + (1 - self.wind_power[col]))
            else:
                row = max(0, row + (1 - self.wind_power[col]))
        if action == 3:  # right
            row = max(0, row - self.wind_power[col])
            col = min(self.col_num - 1, col + 1)
        if action == 4:  # left-up
            row = max(0, row - (1 + self.wind_power[col]))
            col = max(0, col - 1)
        if action == 5:  # left-down
            col = max(0, col - 1)
            if 1 - self.wind_power[col] >= 0:
                row = min(self.row_num - 1, row + (1 - self.wind_power[col]))
            else:
                row = max(0, row + (1 - self.wind_power[col]))
        if action == 6:  # right-up
            row = max(0, row - (1 + self.wind_power[col]))
            col = min(self.col_num - 1, col + 1)
        if action == 7:  # right-down
            if 1 - self.wind_power[col] >= 0:
                row = min(self.row_num - 1, row + (1 - self.wind_power[col]))
            else:
                row = max(0, row + (1 - self.wind_power[col]))
            col = min(self.col_num - 1, col + 1)
        if action == 8:  # stay put
            row = max(0, row - self.wind_power[col])
        self.cur_state = Utils.rowcol2state(row, col, self.col_num)
        if self.cur_state not in self.terminal_states:
            return self.cur_state, -1, False
        else:
            return self.cur_state, -1, True

    def print_path(self):
        for i in range(7):
            for j in range(10):
                if i * 10 + j in self.path:
                    print("[%d]" % self.pi[i * 10 + j], end=' ')
                else:
                    print(" %d " % self.pi[i * 10 + j], end=' ')
            print()
        print()
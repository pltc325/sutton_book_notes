import numpy as np


class PolicyIteration(object):

    def __init__(self, trans_probs, rewards, action_probs, states_num, actions_num, gamma, states_terminal):
        self.trans_probs = trans_probs
        self.rewards = rewards
        self.action_probs = action_probs
        self.states_num = states_num
        self.actions_num = actions_num
        self.v = np.zeros(states_num)
        self.gamma = gamma
        self.states_terminal = states_terminal

    def policy_evaluation(self, diff_threshold = 0.001, iter_max=1000000):
        iter = 0
        while True and iter < iter_max:
            print("iter", iter)
            eval_v_diff = 0
            old_v = self.v.copy()
            for i in range(self.states_num):
                temp = self.v[i]
                new_v = 0
                for j in range(self.actions_num):
                    x = self.action_probs[i][j] * self.trans_probs[i][j]
                    y = self.rewards[i][j] + self.gamma*old_v
                    z = np.dot(x, y)
                    new_v += z
                self.v[i] = new_v
                eval_v_diff = max(eval_v_diff,abs(temp-self.v[i]))
            print(i,"delta",eval_v_diff)
            if eval_v_diff < diff_threshold:
                break
            iter += 1

    def policy_improvement(self, diff_threshold = 0.001, iter_max=1000000):
        is_policy_stable = False
        iter = 0
        while not is_policy_stable and iter < iter_max:
            is_policy_stable = True
            for s in range(self.states_num):
                if s in self.states_terminal:
                    continue
                old_action_prob = self.action_probs[s].copy()
                max_actions = []
                max_q_sa = -1000
                for a in range(self.actions_num):
                    q_sa = 0
                    for s_prime in range(self.states_num):
                        q_sa += self.trans_probs[s][a][s_prime] * (self.rewards[s][a][s_prime] + self.gamma * self.v[s_prime])
                    if q_sa > max_q_sa: # a new max found
                        max_actions = [a]
                        max_q_sa = q_sa
                    elif q_sa == max_q_sa: # a equally new max found, we keep this choice too
                        max_actions.append(a)
                p = 1.0/len(max_actions)
                new_action_prob = np.zeros(self.actions_num)
                for a in range(self.actions_num):
                    if a in max_actions:
                        new_action_prob[a] = p
                max_diff = np.max(np.abs(old_action_prob - new_action_prob))
                if max_diff > diff_threshold:
                    self.action_probs[s] = new_action_prob
                    is_policy_stable = False
                iter += 1

    def policy_iteration(self, eval_diff_threshold = 0.001, imp_diff_eval_diif_threshold = 0.001, iter_diff_threshold = 0.001, iter_max=1000000):
        is_converged = False
        iter = 0
        while not is_converged and iter < iter_max:
            old_v = self.v.copy()
            self.policy_evaluation(eval_diff_threshold, iter_max)
            self.policy_improvement(imp_diff_eval_diif_threshold, iter_max)
            max_diff = np.max(np.abs(self.v - old_v))
            print("max_diff",max_diff)
            if max_diff < iter_diff_threshold:
                is_converged = True
            iter += 1

    def show_v(self, row_num = None, col_num = None):
        if row_num is None and col_num is None:
            for s in range(self.states_num):
                print("%.2f" % self.v[s], end=" ")
        else:
            for i in range(row_num):
                for j in range(col_num):
                    print("%.2f" % self.v[i * col_num + j], end=" ")
                print()
        print()

    def show_policy(self, row_num = None, col_num = None):
        if row_num is None and col_num is None:
            for s in range(self.states_num):
                print("%.2f" % self.v[s], end=" ")
        else:
            for i in range(row_num):
                for j in range(col_num):
                    print(self.action_probs[i * col_num + j], end=" ")
                print()
        print()
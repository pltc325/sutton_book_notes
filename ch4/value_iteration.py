import numpy as np

'''
Policy evaluation in policy iteration is iterative, its convergence is a sufficient condition of optimal policy found(of
course after several iterations), yet it's not a necessary condition, meaning that the optimal policy can be found while
we stop policy evaluation before it is converged. This is considered a drawback of policy iteration.
 
To solve this problem, one way is to after each policy evaluation do the policy improvement right way, and this is 
essentially the value iteration algorithm.
'''


class ValueIteration(object):
    def __init__(self, trans_probs, rewards, action_probs, states_num, actions_num, gamma, states_terminal):
        self.trans_probs = trans_probs
        self.rewards = rewards
        self.action_probs = action_probs
        self.states_num = states_num
        self.actions_num = actions_num
        self.v = np.zeros(states_num)
        self.v[0] = 0
        self.v[100] = 0
        self.gamma = gamma
        self.states_terminal = states_terminal

    def value_iteration(self, diff_threshold = 0.001, iter_max=1000000):
        is_converged = False
        iter = 0
        # find v*
        while not is_converged and iter < iter_max:
            v_old = self.v.copy()
            for s in range(self.states_num):
                vs_max = -1000000
                for a in range(0, self.actions_num):
                    vs_new = np.dot(self.trans_probs[s][a], self.rewards[s][a] + self.gamma*v_old)
                    if vs_new > vs_max:
                        vs_max = vs_new
                self.v[s] = vs_max
            diff = np.max(np.abs(v_old - self.v))
            print("diff",diff)
            if diff < diff_threshold:
                is_converged = True
            iter += 1

        # find pi*
        for s in range(self.states_num):
            if s in self.states_terminal:
                continue
            vs_max = -1000000
            # a can't be started from 0, since this makes s_prime = s, which is max
            for a in range(1, self.actions_num):
                vs_new = np.dot(self.trans_probs[s][a], self.rewards[s][a] + self.gamma*self.v)
                # float version comparison of vs_new > vs_max
                if vs_new - vs_max > 1e-10:
                    max_actions = [a]
                    vs_max = vs_new
                # float version comparison of vs_new == vs_max
                elif abs(vs_new  -  vs_max) < 1e-10:
                    max_actions.append(a)
            p = 1.0 / len(max_actions)
            new_action_prob = np.zeros(self.actions_num)
            for a in range(self.actions_num):
                if a in max_actions:
                    new_action_prob[a] = p
            self.action_probs[s] = new_action_prob

    def show_v(self, row_num = None, col_num = None):
        if row_num is None and col_num is None:
            for s in range(self.states_num):
                print(s,self.v[s])
        else:
            for i in range(row_num):
                for j in range(col_num):
                    print("%.2f" % self.v[i * col_num + j], end=" ")
                print()
        print()

    def show_policy(self, row_num = None, col_num = None):
        if row_num is None and col_num is None:
            for s in range(self.states_num):
                print(self.action_probs[s], end=" ")
        else:
            for i in range(row_num):
                for j in range(col_num):
                    print(self.action_probs[i * col_num + j], end=" ")
                print()
        print()
import numpy as np
from ch4.policy_iteration import  PolicyIteration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Simulation of Jack's rental car problem.

For the number of the requested cars, we exhaust all the possibility up to certain limit(beyond which the
prob is so small that we can safely neglect it).
For the number of the returned cars, we use the expected number(the lambda) as the approximations, it has
two advantages:
1) faster
2) the reward function can be computed at the same time as the transition probability is computed for it only
depends on the number of cars borrowed

The reward function is a little bit difficult to estimate as its parameters s, a, s' don't have                   direct
relation with it. Hence it can't accurately be computed  independently from the policy iteration process,
it needs to be computed during it. This mixes the problem and the algorithm together. What we do here is 
try to approximate the reward function so that it can be computed beforehand.

Also note that we use the integer (state_i * STATE_MAX + state_j), that is [0, 441], to represent
the tuple (state_i, state_j), (0,0) -> (20,20)
and actions are [0, 10], representing [-5,5]
'''
class JacksRentalCarProblem(object):

    def __init__(self, states_num, action_num, lambdas, gamma, is_problem_changed = False):
        self.states_num = states_num
        self.action_num = action_num
        self.trans_probs = np.zeros((states_num*states_num, action_num, states_num*states_num))
        self.rewards = np.zeros((states_num*states_num, action_num, states_num*states_num))
        self.lambdas = lambdas
        self.gamma = gamma
        self.actions = list(range(0,11))
        self.is_problem_changed = is_problem_changed

        poisson_num_upper_bound = 11
        for i in range(states_num):
            for j in range(states_num):
                for a in self.actions:
                    reward = 0
                    if 0< a < 6 and a > i:
                        continue
                    if a >= 6  and a - 5 > j:
                        continue

                    for borrow_a in range(poisson_num_upper_bound):
                        for borrow_b in range(poisson_num_upper_bound):
                            tomorrow_morning_a = i - get_a(a)
                            tomorrow_morning_a = min(20, tomorrow_morning_a)
                            tomorrow_morning_b = j + get_a(a)
                            tomorrow_morning_b = min(20, tomorrow_morning_b)
                            borrow_a_actual = min(borrow_a, tomorrow_morning_a)
                            borrow_b_actual = min(borrow_b, tomorrow_morning_b)
                            cost = 0
                            if is_problem_changed: # the exercise
                                if get_a(a) > 0:
                                    cost += (get_a(a) - 1) * 2
                                else:
                                    cost += get_a(a) * 2
                                if tomorrow_morning_a > 10:
                                    cost += 4
                                if tomorrow_morning_b > 10:
                                    cost += 4
                            else: # original problem
                                cost += get_a(a) * 2
                            reward = (borrow_a_actual + borrow_b_actual) * 10 - cost
                            prob_a = poisson_prob(lamb=self.lambdas['req']['a'], n= borrow_a)
                            prob_b = poisson_prob(lamb=self.lambdas['req']['b'], n= borrow_b)
                            prob = prob_a * prob_b
                            new_i = min(20, tomorrow_morning_a - borrow_a_actual + self.lambdas['ret']['a'])
                            new_j = min(20, tomorrow_morning_b - borrow_b_actual + self.lambdas['ret']['b'])
                            self.trans_probs[i*21+j][a][new_i*21+new_j] += prob
                            # here we approximate reward function, if you want accurate one, it should be better
                            # computed during policy iteration process.
                            # see https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/CarRental.py
                            self.rewards[i*21+j][a][new_i*21+new_j] = reward


def get_a(a):
    """ mapping back from index [0, 10] to [-5, 5]
    :param a: action from [0, 10]
    :return: action from [-5, 5]
    """
    if a == 0:
        return 0
    elif 0 < a < 6:
        return a
    else:
        return 5 - a


def poisson_prob(lamb, n):
    """ return probability of poisson distribution 
    p = lambda^n / n! * exp(-lambda)
    :param lamb: 
    :param n: 
    :return: 
    """
    denominator = 1
    for i in range(n):
        denominator *= (i+1)
    return np.power(lamb, n) / denominator * np.exp(-lamb)

if __name__ == '__main__':
    lambdas = {'req':{'a':3, 'b':4}, 'ret':{'a':3, 'b':2}}
    action_probs = np.ones((21*21,11)) / 11 # every action gets equal prob at beginning
    # original problem
    jrcp = JacksRentalCarProblem(21, 11, lambdas, 0.9, is_problem_changed=False)
    policy_iteration = PolicyIteration( jrcp.trans_probs, jrcp.rewards, action_probs, 21*21, 11, 0.9, [])
    policy_iteration.policy_iteration()
    policy_iteration.show_v(row_num=21,col_num=21)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x = np.arange(21)
    y = np.arange(21)
    x, y = np.meshgrid(x, y)
    v = policy_iteration.v.reshape((21,21))
    ax1.plot_surface(x, y, v)

    # exercise
    jrcp = JacksRentalCarProblem(21, 11, lambdas, 0.9, is_problem_changed=True)
    policy_iteration = PolicyIteration(jrcp.trans_probs, jrcp.rewards, action_probs, 21 * 21, 11, 0.9, [])
    policy_iteration.policy_iteration()
    policy_iteration.show_v(row_num=21, col_num=21)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d', sharex=ax1, sharey=ax1)
    x = np.arange(21)
    y = np.arange(21)
    x, y = np.meshgrid(x, y)
    v = policy_iteration.v.reshape((21,21))
    ax2.plot_surface(x, y, v)

    plt.xlabel("Location A")
    plt.ylabel("Location B")
    plt.show()
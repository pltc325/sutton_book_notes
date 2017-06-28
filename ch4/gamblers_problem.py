import numpy as np
from ch4.value_iteration import ValueIteration
import matplotlib.pyplot as plt

'''
Note since a = 0 doesn't change the state, we exclude it from the investigation. 
'''
class GamblersProblem(object):
    def __init__(self, states_num, action_num, gamma, p):
        self.states_num = states_num
        self.action_num = action_num
        self.trans_probs = np.zeros((states_num, action_num, states_num))
        self.rewards = np.zeros((states_num, action_num, states_num))
        self.gamma = gamma
        self.p = p

        self.actions_of_s = {}
        for s in range(states_num):
            x = min(s, states_num - 1 - s)
            self.actions_of_s[s] = list(range(x+1))

        # for s in range(states_num):
        #     self.actions_of_s[s] = list(range(min(s, states_num-s)+1))
        print(self.actions_of_s)
        for s in range(states_num):
            for a in range(action_num):
                if s not in [0, 100]:
                    if a in self.actions_of_s[s]:
                        if a != 0:
                            #s_prime = min(100,s+a)
                            self.trans_probs[s][a][s+a] = p
                            self.trans_probs[s][a][s-a] = 1 - p
                        else:
                            self.trans_probs[s][a][s] = 1


        for s in range(states_num):
            for a in range(action_num):
                for s_prime in range(states_num):
                    if s_prime == 100:
                        self.rewards[s][a][s_prime] = 1
                    else:
                        self.rewards[s][a][s_prime] = 0


if __name__ == "__main__":
    states_num = 101
    actions_num = 101
    states_terminal = [0,100]
    gp = GamblersProblem(101, 101, 1, 0.51)
    action_probs = np.ones((101,101)) / 101
    action_probs[0] = 0
    action_probs[100] = 0

    value_iteration = ValueIteration(gp.trans_probs, gp.rewards, action_probs, states_num, actions_num, 1, states_terminal)
    value_iteration.value_iteration(diff_threshold=0.000001)
    value_iteration.show_v()
    #value_iteration.show_policy()
    plt.subplot(2,1,1)
    x = np.argmax(value_iteration.action_probs[1:100],axis=1)
    for i in range(99):
        print(i+1,x[i])
    print(14, value_iteration.action_probs[14])
    print(15,value_iteration.action_probs[15])
    print(16,value_iteration.action_probs[16])
    print(18, value_iteration.action_probs[18])
    print(19, value_iteration.action_probs[19])
    print(25, value_iteration.action_probs[25])
    print(36, value_iteration.action_probs[36])
    print(50, value_iteration.action_probs[50])
    print(51, value_iteration.action_probs[51])
    print(52, value_iteration.action_probs[52])
    print(75, value_iteration.action_probs[75])
    print(76, value_iteration.action_probs[76])
    print(80, value_iteration.action_probs[80])
    print(85, value_iteration.action_probs[85])
    print(87, value_iteration.action_probs[87])
    print(88, value_iteration.action_probs[88])
    print(90, value_iteration.action_probs[90])
    print(95, value_iteration.action_probs[95])
    print(96, value_iteration.action_probs[96])
    print(97, value_iteration.action_probs[97])
    print(98, value_iteration.action_probs[98])
    print(99, value_iteration.action_probs[99])

    plt.scatter(range(1,100),x)

    plt.subplot(2,1,2)
    plt.scatter(range(1,100),value_iteration.v[1:100], color='g')
    plt.show()
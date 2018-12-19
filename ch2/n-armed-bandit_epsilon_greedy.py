import numpy as np
import matplotlib.pyplot as plt
from utils import NArmedBandit
from utils import TestBed
'''
This code compares the performances of different
epsilons using epsilon-greedy strategy under stationary
situation.

In case of epsilon = 0, it works worst for both average reward and best hits count because it never explore, so 
it's unlikely for it to find the best action.

Epsilon = 0.1 outperforms epsilon = 0.01 for all the experiments at first some thousands steps, but it is surpassed later.
This is because at early stage it spots good action very quickly since it explore more, but even if it finds the best action,
it will still choose non optimal action for the same reason. To the contrary, epsilon_0.01 find the best action slower, but
once it gets it, it sticks with it, hence gain more value ever after it.
'''


if __name__ == '__main__':
    num_steps = 400
    num_arms = 10
    epsilons = [0.1, 0.01, 0]
    with_noise = False
    bandits = [NArmedBandit("eps:{0}".format(eps), num_arms, eps, num_steps, with_noise) for eps in epsilons]

    num_time = 100
    testbed = TestBed(bandits, num_time=num_time)

    testbed.run()
    testbed.show()
